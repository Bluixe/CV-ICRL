import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import wandb
os.environ.setdefault("WANDB_MODE", "disabled")
from wandb.integration.sb3 import WandbCallback
from tqdm import tqdm

import argparse
import os
import pickle
import json
from pathlib import Path
from e1_minigrid_diagnostics import aggregate_episode_metrics
import random
from collections import deque

import numpy as np
import torch
from IPython import embed
import common_args
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from loguru import logger
import pprint
from procgen import ProcgenEnv
from train_procgen_ppo_official import ProcgenRGBWrapper
from nets.net import device
from nets.impala_cnn import ImpalaCNN
from train_icl_procgen import ProcgenTransformer, ProcgenMultiheadTransformer, PROCGEN_ACTION_DIM

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiProcgenVecEnv(VecEnv):
    """自定义向量化环境，用于合并多个ProcgenRGBWrapper环境"""

    def __init__(self, envs):
        """
        Args:
            envs: 多个ProcgenRGBWrapper环境的列表
        """
        self.envs = envs
        self.num_envs = len(envs)

        # 从第一个环境获取观察空间和动作空间
        observation_space = envs[0].observation_space
        action_space = envs[0].action_space

        super().__init__(self.num_envs, observation_space, action_space)

    def reset(self):
        """重置所有环境"""
        obs_list = []
        for env in self.envs:
            obs = env.reset()
            # ProcgenRGBWrapper返回的是 (num_envs, C, H, W)，但每个env只有1个环境
            # 所以需要去掉batch维度
            if len(obs.shape) == 4:
                obs = obs[0]  # 去掉batch维度
            obs_list.append(obs)
        # 堆叠成 (num_envs, C, H, W)
        return np.stack(obs_list, axis=0)

    def step_async(self, actions):
        """异步执行动作"""
        # 确保actions是numpy数组
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        self.actions = actions

    def step_wait(self):
        """等待步骤完成并返回结果"""
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        for i, env in enumerate(self.envs):
            # 将action转换为numpy数组，确保是int32类型（ProcgenEnv要求）
            action = np.array([self.actions[i]], dtype=np.int32)
            obs, reward, done, info = env.step(action)
            # ProcgenRGBWrapper返回的是 (num_envs, C, H, W)，但每个env只有1个环境
            if len(obs.shape) == 4:
                obs = obs[0]  # 去掉batch维度
            obs_list.append(obs)
            rewards_list.append(reward[0] if isinstance(reward, (list, np.ndarray)) else reward)
            dones_list.append(done[0] if isinstance(done, (list, np.ndarray)) else done)
            infos_list.append(info[0] if isinstance(info, list) else info)

        # 堆叠成向量化格式
        obs = np.stack(obs_list, axis=0)
        rewards = np.array(rewards_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=bool)
        infos = infos_list

        return obs, rewards, dones, infos

    def step(self, actions):
        """同步执行动作（兼容性方法）"""
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        """获取环境属性"""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        """设置环境属性"""
        if indices is None:
            indices = list(range(self.num_envs))
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """调用环境方法"""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        """检查环境是否被指定的wrapper包装"""
        if indices is None:
            indices = list(range(self.num_envs))
        # ProcgenRGBWrapper是VecEnvWrapper，不是标准的Gym wrapper
        # 对于VecEnvWrapper，通常返回False，因为它们不是标准的Gym wrapper
        # 如果需要检查底层环境，可以调用envs[i].env_is_wrapped(wrapper_class)
        results = []
        for i in indices:
            env = self.envs[i]
            # 如果env有env_is_wrapped方法（VecEnvWrapper通常有），调用它
            if hasattr(env, 'env_is_wrapped'):
                try:
                    # VecEnvWrapper的env_is_wrapped返回列表，取第一个元素
                    wrapped = env.env_is_wrapped(wrapper_class)
                    results.append(wrapped[0] if isinstance(wrapped, list) and len(wrapped) > 0 else False)
                except:
                    results.append(False)
            else:
                results.append(False)
        return results

class ContextBuffer:
    """用于存储最近的交互历史的缓冲区"""

    def __init__(self, horizon, state_dim, action_dim, use_values=False):
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_values = use_values  # 是否使用context_values

        # 初始化缓冲区
        self.states = deque(maxlen=horizon)
        self.actions = deque(maxlen=horizon-1)
        self.rewards = deque(maxlen=horizon-1)
        if self.use_values:
            self.values = deque(maxlen=horizon-1)  # 用于存储context_values

    def add(self, state, action, reward, value=None, first_step=False):
        """添加一个新的交互到缓冲区"""
        if first_step:
            self.states.append(state)
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            if self.use_values:
                if value is not None:
                    self.values.append(value)
                else:
                    self.values.append(0.0)  # 如果没有提供value，使用0

    def get_context(self):
        """获取当前缓冲区中的所有交互作为上下文"""
        # 如果缓冲区未满，用零填充
        padding_size = self.horizon - len(self.actions)

        if len(self.states) == 1:
            # 如果缓冲区为空，创建全零数组
            context_states = np.array([self.states[0]])  # [1, C, H, W]
            context_actions = np.zeros((1,1))
            context_rewards = np.zeros((1,1))
            if self.use_values:
                context_values = np.zeros((1,1))
        elif padding_size > 0:
            # 创建填充
            padding_actions = np.zeros(1)
            padding_rewards = np.zeros(1)
            if self.use_values:
                padding_values = np.zeros(1)

            # 将deque中的元素转换为numpy数组，确保保持正确的形状
            context_states = np.stack(list(self.states))
            context_actions = np.concatenate([np.array(list(self.actions)), padding_actions])
            context_rewards = np.concatenate([np.array(list(self.rewards)), padding_rewards])
            if self.use_values:
                context_values = np.concatenate([np.array(list(self.values)), padding_values])
        else:
            # 如果缓冲区已满，直接使用所有数据
            padding_actions = np.zeros(1)
            padding_rewards = np.zeros(1)
            if self.use_values:
                padding_values = np.zeros(1)

            context_states = np.stack(list(self.states))
            context_actions = np.concatenate([np.array(list(self.actions)), padding_actions])
            context_rewards = np.concatenate([np.array(list(self.rewards)), padding_rewards])
            if self.use_values:
                context_values = np.concatenate([np.array(list(self.values)), padding_values])

        # 转换为张量
        # Procgen图像已经是 (C, H, W) 格式，uint8
        context_states = torch.tensor(context_states, dtype=torch.uint8).to(device)
        context_actions = torch.tensor(context_actions, dtype=torch.long).to(device)
        context_rewards = torch.tensor(context_rewards, dtype=torch.float32).to(device)

        context_states = context_states.unsqueeze(0)  # [1, seq_len, C, H, W]

        # context_actions 应该是 [batch_size, seq_len]
        if len(context_actions.shape) == 1:  # [seq_len]
            context_actions = context_actions.unsqueeze(0)  # [1, seq_len]

        # context_rewards 应该是 [batch_size, seq_len, 1]
        if len(context_rewards.shape) == 1:  # [seq_len]
            context_rewards = context_rewards.unsqueeze(-1)  # [seq_len, 1]
            context_rewards = context_rewards.unsqueeze(0)  # [1, seq_len, 1]

        result = {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_rewards': context_rewards,
        }

        # 如果使用values，添加到结果中
        if self.use_values:
            context_values_tensor = torch.tensor(context_values, dtype=torch.float32).to(device)
            if len(context_values_tensor.shape) == 1:  # [seq_len]
                context_values_tensor = context_values_tensor.unsqueeze(-1)  # [seq_len, 1]
                context_values_tensor = context_values_tensor.unsqueeze(0)  # [1, seq_len, 1]
            result['context_values'] = context_values_tensor

        return result

def preprocess_obs(obs):
    """预处理观察，ProcgenRGBWrapper已经将观察转换为(C, H, W)格式"""
    # ProcgenRGBWrapper已经将观察转换为channel-first格式
    # obs shape: (C, H, W) 或 (num_envs, C, H, W)
    # 确保是uint8类型
    if obs.dtype != np.uint8:
        obs = obs.astype(np.uint8)
    return obs

def make_procgen_env(env_name, num_envs, seed_list=None, num_levels=0, start_level=0, distribution_mode="hard"):
    """创建多个并行Procgen环境

    如果提供了有效的seed_list（所有值都非-1），则为每个环境创建独立的环境，
    每个环境使用seed_list[i]作为start_level，num_levels=1，确保每个环境固定在一个关卡上。
    否则，创建单个向量化环境，所有环境共享相同的start_level和num_levels。
    """
    if seed_list is None:
        seed_list = [-1] * num_envs
    assert num_envs == len(seed_list), "Number of environments must match number of seeds"

    # 如果提供了有效的seed_list（所有值都非-1），为每个环境创建独立的环境，每个环境固定在一个关卡上
    if all(s != -1 for s in seed_list):
        # 为每个环境创建独立的环境，每个环境固定在一个关卡上
        envs = []
        for i in range(num_envs):
            print(f"[ENV SETUP] Creating Env {i} with start_level={seed_list[i]}, num_levels=1")
            env = ProcgenEnv(
                num_envs=1,
                env_name=env_name,
                num_levels=1,  # 每个环境只有1个关卡
                start_level=seed_list[i],  # 第i个环境使用关卡 seed_list[i]
                distribution_mode=distribution_mode
            )
            # 提取RGB观察并转换为channel-first格式
            env = ProcgenRGBWrapper(env, "rgb")
            envs.append(env)

        # 使用自定义的向量化环境合并多个环境
        env = MultiProcgenVecEnv(envs)
        print(f"[ENV SETUP] Created {num_envs} environments with fixed levels: {seed_list}")
        return env

    # 否则，创建单个向量化环境（原有逻辑）
    env = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode
    )
    # 提取RGB观察并转换为channel-first格式
    env = ProcgenRGBWrapper(env, "rgb")

    return env

def eval_models_multi_seed(model, env, horizon, num_steps=2000, relabel_reward=False, wo_reward=False, auto_relabel=False, output_json=None):
    """并行评估ICL模型在多个Procgen环境中的表现"""

    num_envs = env.num_envs

    model_name = model.__class__.__name__

    # 为每个环境创建上下文缓冲区
    # Procgen图像是 (C, H, W) = (3, 64, 64)
    state_dim = 3 * 64 * 64  # 占位符，实际使用图像
    action_dim = env.action_space.n
    # 如果使用auto_relabel，需要维护context_values
    context_buffers = [ContextBuffer(horizon, state_dim, action_dim, use_values=auto_relabel) for _ in range(num_envs)]

    total_rewards = np.zeros(num_envs)
    episode_counts = np.zeros(num_envs)
    episode_returns = [[] for _ in range(num_envs)]
    unfinished = np.zeros(num_envs)

    # 用于跟踪每个环境的 level_seed，以验证是否意外切换
    level_seed_history = {}  # {env_idx: [list of level_seeds]}
    for env_idx in range(num_envs):
        level_seed_history[env_idx] = []

    # 重置环境
    obs = env.reset()
    # 预处理观察
    # ProcgenRGBWrapper已经返回 (num_envs, C, H, W) 格式
    obs = preprocess_obs(obs)

    # 尝试在重置后获取初始 level_seed 信息
    try:
        if hasattr(env, 'envs'):
            # MultiProcgenVecEnv: 从每个子环境获取
            for env_idx, sub_env in enumerate(env.envs):
                try:
                    if hasattr(sub_env, 'venv') and hasattr(sub_env.venv, 'get_info'):
                        raw_info = sub_env.venv.get_info()
                        if isinstance(raw_info, list) and len(raw_info) > 0:
                            if isinstance(raw_info[0], dict) and 'level_seed' in raw_info[0]:
                                initial_seed = raw_info[0]['level_seed']
                                level_seed_history[env_idx].append(initial_seed)
                                print(f"[INIT] Env {env_idx} initial level_seed={initial_seed}")
                except Exception as e:
                    print(f"[INIT] Could not get initial level_seed for Env {env_idx}: {e}")
    except Exception as e:
        print(f"[INIT] Could not get initial level_seed: {e}")

    for i in range(num_envs):
        # 初始化上下文缓冲区
        context_buffers[i].add(obs[i], 0, 0, first_step=True)

    # 打印观察的形状，用于调试
    logger.info(f"观察形状: {obs.shape}")

    dones = np.array([False] * num_envs)
    relabel_rewards = np.zeros(num_envs, dtype=np.float32)
    labels = np.zeros(num_envs, dtype=np.float32)  # 用于存储ProcgenMultiheadTransformer的pred_values
    current_values = np.zeros(num_envs, dtype=np.float32)  # 用于存储当前的context_values
    for t in tqdm(range(num_steps), desc="评估进度", ncols=100):
        # 为每个环境准备上下文和动作
        actions = np.zeros(num_envs, dtype=np.int64)  # 默认动作为0

        for i in range(num_envs):
            # 获取上下文
            context = context_buffers[i].get_context()
            context_len = context['context_states'].shape[1]

            # 使用模型预测动作
            with torch.no_grad():
                if model_name == 'ProcgenTransformer':
                    action_probs = model(context)
                elif model_name == 'ProcgenMultiheadTransformer':
                    # 如果使用auto_relabel，需要传入context_values
                    if auto_relabel:
                        action_probs, pred_values = model(context, auto_relabel=True, pred_actions=True)
                    else:
                        action_probs, pred_values = model(context, auto_relabel=False, pred_actions=True)
                else:
                    raise ValueError(f"Unsupported model type: {model_name}")

                # 如果模型输出是logits，转换为概率
                if len(action_probs.shape) == 3:
                    action_probs = action_probs[0, context_len-1]  # 取最后一个时间步的预测
                    if model_name == 'ProcgenMultiheadTransformer':
                        pred_values = pred_values[0, context_len-1]

                # 采样动作
                action = torch.multinomial(torch.softmax(action_probs, dim=-1), num_samples=1).item()
                actions[i] = action
                if model_name == 'ProcgenMultiheadTransformer':
                    labels[i] = pred_values.item()
                    # 更新当前的value（用于下一个step的context_values）
                    if auto_relabel:
                        current_values[i] = pred_values.item()

        # 执行动作
        next_obs, rewards, dones, infos = env.step(actions)

        # 提取并记录 level_seed 信息以验证是否切换
        level_seeds = [None] * num_envs

        if isinstance(infos, list):
            # 向量化环境：infos 是列表，每个元素对应一个环境
            for env_idx, env_info in enumerate(infos):
                if isinstance(env_info, dict):
                    level_seed = env_info.get('level_seed', None)
                    level_seeds[env_idx] = level_seed
        elif isinstance(infos, dict):
            # 单个环境：infos 是字典
            level_seed = infos.get('level_seed', None)
            level_seeds[0] = level_seed

        # 如果 infos 中没有 level_seed，尝试从底层环境获取
        if all(ls is None for ls in level_seeds):
            try:
                # 尝试从底层 ProcgenEnv 获取 info
                if hasattr(env, 'venv') and hasattr(env.venv, 'get_info'):
                    raw_info = env.venv.get_info()
                    if isinstance(raw_info, list):
                        for env_idx, raw_env_info in enumerate(raw_info):
                            if isinstance(raw_env_info, dict) and 'level_seed' in raw_env_info:
                                level_seeds[env_idx] = raw_env_info['level_seed']
                    elif isinstance(raw_info, dict) and 'level_seed' in raw_info:
                        level_seeds[0] = raw_info['level_seed']
                # 对于 MultiProcgenVecEnv，尝试从每个子环境获取
                elif hasattr(env, 'envs'):
                    for env_idx, sub_env in enumerate(env.envs):
                        try:
                            if hasattr(sub_env, 'venv') and hasattr(sub_env.venv, 'get_info'):
                                raw_info = sub_env.venv.get_info()
                                if isinstance(raw_info, list) and len(raw_info) > 0:
                                    if isinstance(raw_info[0], dict) and 'level_seed' in raw_info[0]:
                                        level_seeds[env_idx] = raw_info[0]['level_seed']
                                elif isinstance(raw_info, dict) and 'level_seed' in raw_info:
                                    level_seeds[env_idx] = raw_info['level_seed']
                        except Exception as e:
                            pass
            except Exception as e:
                logger.debug(f"Failed to get level_seed from underlying env: {e}")

        # 记录 level_seed 信息并输出验证信息
        for env_idx, level_seed in enumerate(level_seeds):
            if level_seed is not None:
                # 检查是否切换了 level_seed
                if len(level_seed_history[env_idx]) > 0:
                    prev_level_seed = level_seed_history[env_idx][-1]
                    if level_seed != prev_level_seed:
                        print(f"[WARNING] Level seed changed! Step {t}, Env {env_idx}: "
                              f"prev_level_seed={prev_level_seed}, new_level_seed={level_seed}, done={dones[env_idx]}")
                level_seed_history[env_idx].append(level_seed)

                # 每10步输出一次 level_seed 信息（避免日志过多）
            #     if t % 10 == 0 or dones[env_idx]:
            #         print(f"[VERIFY] Step {t}, Env {env_idx}: level_seed={level_seed}, done={dones[env_idx]}")
            # elif t == 0:  # 只在第一步时输出警告
            #     print(f"[WARNING] Could not extract level_seed for Env {env_idx} at step {t}")

        if relabel_reward:
            if model_name == 'ProcgenMultiheadTransformer':
                # relabel_rewards = np.maximum(relabel_rewards, labels)
                # logger.info(f"labels: {labels}")
                relabel_rewards = labels
            else:
                # 如果需要重新标记reward，可以在这里实现
                relabel_rewards = np.random.random(rewards.shape)

        # 预处理下一个观察
        # ProcgenRGBWrapper已经返回 (num_envs, C, H, W) 格式
        next_obs = preprocess_obs(next_obs)

        # 更新上下文缓冲区和统计信息
        for i in range(num_envs):
            if auto_relabel:
                # 如果使用auto_relabel，需要传入value
                if relabel_reward:
                    context_buffers[i].add(next_obs[i], actions[i], relabel_rewards[i], value=current_values[i])
                elif wo_reward:
                    context_buffers[i].add(next_obs[i], actions[i], 0, value=current_values[i])
                else:
                    context_buffers[i].add(next_obs[i], actions[i], rewards[i], value=current_values[i])
            else:
                if relabel_reward:
                    context_buffers[i].add(next_obs[i], actions[i], relabel_rewards[i])
                elif wo_reward:
                    context_buffers[i].add(next_obs[i], actions[i], 0)
                else:
                    context_buffers[i].add(next_obs[i], actions[i], rewards[i])
            total_rewards[i] += rewards[i]
            unfinished[i] += rewards[i]
            if dones[i]:  # 如果环境刚刚完成
                episode_counts[i] += 1
                episode_returns[i].append(float(unfinished[i]))
                unfinished[i] = 0
                logger.info(f"环境 {i} 完成了一个episode，奖励: {total_rewards[i]}")
                # 重置value（episode结束时）
                if auto_relabel:
                    current_values[i] = 0.0

        # 更新观察
        obs = next_obs

        if (t+1) % 1 == 0:
            res_dict1 = {
                f"env_{i}_culmulative reward": total_rewards[i] for i in range(num_envs)
            }
            res_dict2 = {
                f"env_{i}_episode avg reward": total_rewards[i]/episode_counts[i] if episode_counts[i] > 0 else 0.0 for i in range(num_envs)
            }
            time_dict = {"culmulative reward": sum(total_rewards),
                   "episode avg reward": sum(total_rewards)/sum(episode_counts) if sum(episode_counts) > 0 else 0.0}
            wandb.log({**res_dict1, **res_dict2, **time_dict})
        else:
            wandb.log({"culmulative reward": sum(total_rewards),
                   "episode avg reward": sum(total_rewards)/sum(episode_counts) if sum(episode_counts) > 0 else 0.0})

    # 计算平均奖励
    valid_episodes = episode_counts > 0
    if np.any(valid_episodes):
        avg_rewards = total_rewards[valid_episodes] / episode_counts[valid_episodes]
        avg_reward = np.mean(avg_rewards)
    else:
        avg_reward = 0.0

    logger.info(f"Total episodes completed: {np.sum(episode_counts)}")
    logger.info(f"Average reward per episode: {avg_reward}")

    # 输出 level_seed 统计信息
    print("\n=== Level Seed Statistics ===")
    for env_idx in range(num_envs):
        if len(level_seed_history[env_idx]) > 0:
            unique_seeds = set(level_seed_history[env_idx])
            print(f"Env {env_idx}: Total steps={len(level_seed_history[env_idx])}, "
                  f"Unique level_seeds={len(unique_seeds)}, Seeds={sorted(unique_seeds)[:20]}")  # 只显示前20个
            if len(unique_seeds) > 1:
                print(f"[WARNING] Env {env_idx}: Level seed changed {len(unique_seeds)-1} times during evaluation!")
        else:
            print(f"Env {env_idx}: No level_seed information collected")
    print("=" * 30 + "\n")

    if output_json:
        path = Path(output_json); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate_episode_metrics(episode_returns), indent=2, allow_nan=False))
    return avg_reward

def main():
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)

    parser.add_argument('--algorithm', type=str, default='ad', choices=['ad', 'dpt'], help='算法类型')
    parser.add_argument('--num_envs', type=int, default=8, help='并行环境数量')
    parser.add_argument('--num_steps', type=int, default=1000, help='评估的步数')
    parser.add_argument('--fix_seed', action='store_true', help='是否固定随机种子')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--use_value', action='store_true', default=False,
                    help='Use value function or not')
    parser.add_argument('--use_finetune', action='store_true', default=False,
                        help='Use finetune data or not')
    parser.add_argument('--multi_env', action='store_true', default=False,
                        help='Use multiple environments or not')
    parser.add_argument('--eval_env', type=str, default='coinrun',
                        help='Evaluation environment name')
    parser.add_argument('--adjust_loss', action='store_true', default=False,
                        help='Adjust loss for value function or not')
    parser.add_argument('--reward_gain', action='store_true', default=False,
                        help='Use reward gain for value function or not')
    parser.add_argument('--suffix_reward', action='store_true', default=False,
                        help='Use suffix reward for value function or not')
    parser.add_argument('--pred_reward', action='store_true', default=False,
                        help='Use reward for training or not')
    parser.add_argument('--reward_mode', type=int, default=0)
    parser.add_argument('--wo_reward', action='store_true', default=False,
                        help='Without reward for training or not')
    parser.add_argument('--relabel_reward', action='store_true', default=False,
                        help='Relabel reward for training or not')
    parser.add_argument('--auto_relabel', action='store_true', default=False,
                        help='Use automatic relabel for training or not')
    parser.add_argument('--num_levels', type=int, default=0, help='Number of levels (0 for infinite)')
    parser.add_argument('--start_level', type=int, default=10000, help='Start level seed')
    parser.add_argument('--distribution_mode', type=str, default='hard',
                        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'],
                        help='Distribution mode for Procgen')
    parser.add_argument('--model_epoch', type=str, default='15',
                        help='Model epoch to load (e.g., "final", "20", "15"). Default: "final"')

    parser.add_argument('--checkpoint')
    parser.add_argument('--output-json')
    args = vars(parser.parse_args())
    if args['multi_env']: parser.error('This release supports single-task Procgen AD/CV only.')
    if args['output_json'] and Path(args['output_json']).exists(): raise FileExistsError(args['output_json'])

    # 设置随机种子
    seed = args['seed']
    # 为每个环境设置固定的关卡：第i个环境使用关卡 start_level + i
    # 这样如果 start_level=0, num_envs=20，则环境使用关卡 0, 1, 2, ..., 19
    seed_list = [args['start_level'] + i for i in range(args['num_envs'])]
    env_name = args['env']
    algorithm = args['algorithm']
    use_value = args['use_value']
    pred_reward = args['pred_reward']
    reward_mode = args['reward_mode']
    wo_reward = args['wo_reward']
    relabel_reward = args['relabel_reward']
    auto_relabel = args['auto_relabel']
    model_epoch = args.get('model_epoch', 'final')
    multi_env = args.get('multi_env', False)

    # 创建环境
    eval_env_name = args["eval_env"]
    print(f"[MAIN] Creating {args['num_envs']} environments with seed_list={seed_list}")
    print(f"[MAIN] start_level={args['start_level']}, num_levels={args['num_levels']}")
    env = make_procgen_env(
        eval_env_name,
        args['num_envs'],
        seed_list,
        num_levels=args['num_levels'],
        start_level=args['start_level'],
        distribution_mode=args['distribution_mode']
    )
    print(f"[MAIN] Environment created successfully")

    # 设置模型配置
    config = {
        'horizon': args['H'],
        'state_dim': 2,  # 占位符，实际使用图像
        'action_dim': PROCGEN_ACTION_DIM,
        'n_layer': args['layer'],
        'n_embd': args['embd'],
        'n_head': args['head'],
        'shuffle': args['shuffle'],
        'dropout': args['dropout'],
        'test': True,
        'store_gpu': False,  # Procgen图像较大，不存储在GPU上
        'image_shape': (3, 64, 64),  # Procgen RGB图像尺寸 (C, H, W)
    }
    logger.info(config)

    env_name_1 = env_name.replace("procgen:", "").replace("-v0", "")
    eval_env_name_1 = eval_env_name.replace("procgen:", "").replace("-v0", "")

    # 构建wandb名称
    if not use_value and not args['use_finetune'] and not args['multi_env'] and not pred_reward:
        if args['suffix_reward']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-suffix_reward"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-suffix_reward"
        elif args['reward_gain']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-reward_gain"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-reward_gain"
        elif args['wo_reward']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-wo_reward"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-wo_reward"
        elif args['relabel_reward']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-relabel_reward"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-relabel_reward"
        elif args['auto_relabel']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-auto_relabel"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-auto_relabel"
        else:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}"
    elif pred_reward:
        wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-pred_reward_{reward_mode}"
        wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-pred_reward_{reward_mode}"
    elif args['use_finetune']:
        wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-finetune"
        wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-finetune"
    elif args['multi_env']:
        if args['relabel_reward']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_relabel_reward"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_relabel_reward"
        elif args['auto_relabel']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_auto_relabel"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_auto_relabel"
        else:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env"
    elif use_value:
        if args['adjust_loss']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-value-adjust"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-value-adjust"
        else:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-value"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-value"
    else:
        wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}"
        wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}"

    run = wandb.init(
        project="eval-procgen-icl",
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        )

    # 加载模型
    if algorithm == 'ad':
        if use_value or pred_reward:
            if use_value:
                if args['adjust_loss']:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch15_value_adjust.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch15_value.pt'
            else:
                model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch15_pred_reward_{reward_mode}.pt'
            # 注意：目前ProcgenTransformer不支持value预测，需要创建ProcgenMultiheadTransformer
            model = ProcgenTransformer(config, mode="ad").to(device)
        elif args['use_finetune']:
            if model_epoch == 'final':
                model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_finetune_more.pt'
            else:
                model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_finetune_more.pt'
            model = ProcgenTransformer(config, mode="ad").to(device)
        elif args['multi_env']:
            if args['relabel_reward']:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_multi_env_relabel_reward.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_multi_env_relabel_reward.pt'
            elif args['auto_relabel']:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_multi_env_auto_relabel.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_multi_env_auto_relabel.pt'
                model = ProcgenMultiheadTransformer(config, mode="ad").to(device)
            else:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_multi_env.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_multi_env.pt'
                model = ProcgenTransformer(config, mode="ad").to(device)
        else:
            if args['reward_gain']:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_reward_gain.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_reward_gain.pt'
            elif args['suffix_reward']:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_suffix.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_suffix.pt'
            elif args['wo_reward']:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_wo_reward.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_wo_reward.pt'
            elif args['relabel_reward']:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_relabel_reward.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_relabel_reward.pt'
            elif args['auto_relabel']:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_auto_relabel.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_auto_relabel.pt'
                model = ProcgenMultiheadTransformer(config, mode="ad").to(device)
            else:
                if model_epoch == 'final':
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_more.pt'
                else:
                    model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_more.pt'
                model = ProcgenTransformer(config, mode="ad").to(device)
    elif algorithm == 'dpt':
        if model_epoch == 'final':
            model_path = f'models/Procgen/{args["env"]}/{algorithm}/final_dpt.pt'
        else:
            model_path = f'models/Procgen/{args["env"]}/{algorithm}/epoch{model_epoch}_dpt.pt'
        model = ProcgenTransformer(config, mode="ad").to(device)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    logger.info(f"Loading model from: {model_path}")
    model_path = args['checkpoint'] or model_path
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logger.info(f"Evaluating model on {env_name} with {args['algorithm']} mode")
    logger.info(f"Using {args['num_envs']} parallel environments")

    # 评估模型
    avg_reward = eval_models_multi_seed(
        model,
        env,
        horizon=args['H'],
        num_steps=args['num_steps'],
        relabel_reward=(relabel_reward or auto_relabel),
        auto_relabel=auto_relabel,
        output_json=args['output_json'],
    )

    # 记录结果
    results = {
        'env': env_name,
        'algorithm': args['algorithm'],
        'avg_reward': avg_reward,
    }

    # 打印结果
    logger.info("Evaluation Results:")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Algorithm: {args['algorithm']}")
    logger.info(f"Average Reward: {avg_reward}")

    env.close()

    return results

if __name__ == '__main__':
    main()
