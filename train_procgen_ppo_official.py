"""
官方训练方法的PyTorch移植版本
基于train-procgen/train_procgen/train.py，使用stable-baselines3和Impala CNN

保持与官方TensorFlow版本相同的超参数和网络结构
"""
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnvStepReturn
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

import wandb
from wandb.integration.sb3 import WandbCallback
from procgen import ProcgenEnv

import argparse
import os
import random

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from loguru import logger
from nets.impala_cnn import ImpalaCNN
from stable_baselines3.common.callbacks import CheckpointCallback


class ProcgenRGBWrapper(VecEnvWrapper):
    """
    自定义wrapper来提取procgen环境的RGB观察
    将channel-last格式转换为channel-first格式以兼容stable-baselines3
    """
    def __init__(self, venv, key="rgb"):
        self.key = key
        self.render_mode = None  # 设置render_mode属性以避免警告

        # 获取RGB观察空间
        if hasattr(venv.observation_space, 'spaces') and key in venv.observation_space.spaces:
            observation_space = venv.observation_space.spaces[key]
        else:
            # 如果无法获取，尝试从实际观察中推断
            obs = venv.reset()
            if isinstance(obs, dict) and key in obs:
                obs_shape = obs[key].shape[1:]  # 去掉batch维度
                observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
            else:
                raise ValueError(f"Cannot extract observation space for key '{key}'")

        # 将channel-last格式转换为channel-first格式 (H, W, C) -> (C, H, W)
        is_box = (hasattr(observation_space, 'shape') and
                  hasattr(observation_space, 'low') and
                  hasattr(observation_space, 'high'))
        if is_box and len(observation_space.shape) == 3:
            height, width, channels = observation_space.shape
            # 提取low和high值
            if hasattr(observation_space.low, 'min'):
                low_val = float(observation_space.low.min())
            elif np.isscalar(observation_space.low):
                low_val = float(observation_space.low)
            else:
                low_val = float(np.asarray(observation_space.low).flatten()[0])

            if hasattr(observation_space.high, 'max'):
                high_val = float(observation_space.high.max())
            elif np.isscalar(observation_space.high):
                high_val = float(observation_space.high)
            else:
                high_val = float(np.asarray(observation_space.high).flatten()[0])

            observation_space = spaces.Box(
                low=low_val,
                high=high_val,
                shape=(channels, height, width),
                dtype=observation_space.dtype
            )
            self.needs_transpose = True
        else:
            self.needs_transpose = False

        # 确保动作空间是标准的gymnasium类型
        if hasattr(venv.action_space, 'n'):
            action_space = spaces.Discrete(venv.action_space.n)
        else:
            action_space = venv.action_space

        super().__init__(venv=venv, observation_space=observation_space, action_space=action_space)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        if isinstance(obs, dict):
            obs = obs[self.key]

        # 转换为channel-first格式 (num_envs, H, W, C) -> (num_envs, C, H, W)
        if self.needs_transpose and len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))

        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, infos = self.venv.step_wait()

        if isinstance(obs, dict):
            # 处理terminal_observation
            for info in infos:
                if "terminal_observation" in info and isinstance(info["terminal_observation"], dict):
                    term_obs = info["terminal_observation"][self.key]
                    # 转换terminal_observation为channel-first格式 (H, W, C) -> (C, H, W)
                    if self.needs_transpose and len(term_obs.shape) == 3:
                        term_obs = np.transpose(term_obs, (2, 0, 1))
                    info["terminal_observation"] = term_obs
            obs = obs[self.key]

        # 格式转换
        if self.needs_transpose and len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))

        return obs, reward, done, infos


class ProcgenEvalWrapper(VecEnvWrapper):
    """
    包装器，为procgen环境添加env_is_wrapped方法，使其与stable-baselines3的evaluate_policy兼容
    """
    def __init__(self, venv):
        super().__init__(venv, observation_space=venv.observation_space, action_space=venv.action_space)
        self.render_mode = None  # 设置render_mode属性以避免警告

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            return [False] * self.num_envs
        else:
            return [False] * len(indices)


class RewardWandbCallback(WandbCallback):
    def _on_step(self) -> bool:
        # 获取 Monitor 收集的奖励等信息
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                episode_info = info["episode"]
                wandb.log({
                    "train/episode_reward": episode_info["r"],
                    "train/episode_length": episode_info["l"],
                })
        return super()._on_step()


def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level,
             timesteps_per_proc, log_dir='/tmp/procgen', save_path=None, device='cuda:0'):
    """
    训练函数，使用与官方相同的超参数

    :param env_name: 环境名称
    :param num_envs: 并行环境数量
    :param distribution_mode: 分布模式 ("easy", "hard", "exploration", "memory", "extreme")
    :param num_levels: 关卡数量 (0表示无限)
    :param start_level: 起始关卡种子
    :param timesteps_per_proc: 每个进程的总时间步数
    :param log_dir: 日志目录
    :param save_path: 模型保存路径
    :param device: 设备名称 ('cuda:0' 或 'cuda:1')
    """
    # 官方超参数（与train-procgen/train_procgen/train.py保持一致）
    learning_rate = 5e-4
    ent_coef = 0.01
    gamma = 0.999
    lam = 0.95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = 0.2
    use_vf_clipping = True
    vf_coef = 0.5
    max_grad_norm = 0.5

    logger.info("创建环境...")
    # 创建procgen环境
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode
    )
    # 提取RGB观察
    venv = ProcgenRGBWrapper(venv, "rgb")
    # 添加Monitor（用于记录episode统计信息）
    venv = VecMonitor(venv=venv, filename=None)
    # 添加VecNormalize（只归一化奖励，不归一化观察）
    venv = VecNormalize(venv=venv, norm_obs=False, norm_reward=True)
    # 添加eval wrapper
    venv = ProcgenEvalWrapper(venv)

    logger.info("创建PPO模型...")
    # 创建Impala CNN特征提取器（与官方相同：depths=[16,32,32], emb_size=256）
    policy_kwargs = dict(
        features_extractor_class=ImpalaCNN,
        features_extractor_kwargs=dict(depths=[16, 32, 32], emb_size=256),
        net_arch=[],  # 官方使用空列表，表示只有特征提取器，没有额外的MLP层
        activation_fn=None,  # 使用默认激活函数
    )

    # 创建PPO模型，使用官方超参数
    model = PPO(
        "CnnPolicy",
        venv,
        learning_rate=learning_rate,
        n_steps=nsteps,
        batch_size=nsteps * num_envs // nminibatches,  # 总batch size
        n_epochs=ppo_epochs,
        gamma=gamma,
        gae_lambda=lam,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir if log_dir else None,
        device=device,
    )

    logger.info("开始训练...")
    # 创建callbacks
    callbacks = []

    # Wandb callback
    if wandb.run is not None:
        callbacks.append(RewardWandbCallback())

    # Checkpoint callback
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        # 计算save_freq: 想保存10次checkpoint，每次保存间隔 timesteps_per_proc // 10 个时间步
        # 由于使用向量化环境，每次env.step()对应num_envs个时间步
        # 所以需要将时间步数除以num_envs来得到env.step()的调用次数
        save_freq_timesteps = max(timesteps_per_proc // 100, 1)  # 每0.5M时间步保存一次（共100次）
        save_freq = max(save_freq_timesteps // num_envs, 1)  # 转换为env.step()调用次数
        logger.info(f"Checkpoint保存频率: 每 {save_freq_timesteps} 个时间步（约 {save_freq} 次 env.step() 调用）")
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix="checkpoint",
            verbose=2,  # 打印保存信息
        )
        callbacks.append(checkpoint_callback)

    # 训练
    model.learn(
        total_timesteps=timesteps_per_proc,
        callback=callbacks if callbacks else None,
        log_interval=1,
    )

    # 保存最终模型
    if save_path:
        final_model_path = os.path.join(save_path, "final_model")
        model.save(final_model_path)
        logger.info(f"模型已保存到: {final_model_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='官方训练方法的PyTorch移植版本')
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='环境名称')
    parser.add_argument('--num_envs', type=int, default=64,
                        help='并行环境数量')
    parser.add_argument('--distribution_mode', type=str, default='hard',
                        choices=["easy", "hard", "exploration", "memory", "extreme"],
                        help='分布模式')
    parser.add_argument('--num_levels', type=int, default=0,
                        help='关卡数量 (0表示无限)')
    parser.add_argument('--start_level', type=int, default=0,
                        help='起始关卡种子')
    parser.add_argument('--timesteps_per_proc', type=int, default=50_000_000,
                        help='每个进程的总时间步数')
    parser.add_argument('--log_dir', type=str, default='outputs/procgen-ppo',
                        help='日志目录')
    parser.add_argument('--save_path', type=str, default=None,
                        help='模型保存路径')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用wandb记录')
    parser.add_argument('--wandb_project', type=str, default='procgen-ppo-official',
                        help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='wandb运行名称')
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help='设备名称 (cuda:0 或 cuda:1)')

    args = parser.parse_args()

    # 如果没有指定save_path，使用默认路径
    if args.save_path is None:
        args.save_path = f"models/Procgen/{args.env_name}/official"
        logger.info(f"使用默认保存路径: {args.save_path}")

    # 初始化wandb（如果启用）
    if args.use_wandb:
        wandb_name = args.wandb_name or f"procgen-{args.env_name}-official"
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config={
                'env_name': args.env_name,
                'num_envs': args.num_envs,
                'distribution_mode': args.distribution_mode,
                'num_levels': args.num_levels,
                'start_level': args.start_level,
                'timesteps_per_proc': args.timesteps_per_proc,
                'device': args.device,
                # 官方超参数
                'learning_rate': 5e-4,
                'ent_coef': 0.01,
                'gamma': 0.999,
                'lam': 0.95,
                'nsteps': 256,
                'nminibatches': 8,
                'ppo_epochs': 3,
                'clip_range': 0.2,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'impala_depths': [16, 32, 32],
                'impala_emb_size': 256,
            },
            sync_tensorboard=True,
        )

    # 训练
    model = train_fn(
        env_name=args.env_name,
        num_envs=args.num_envs,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        timesteps_per_proc=args.timesteps_per_proc,
        log_dir=args.log_dir,
        save_path=args.save_path,
        device=args.device,
    )

    logger.info("训练完成！")


if __name__ == '__main__':
    main()
