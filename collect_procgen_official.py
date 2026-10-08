from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnvStepReturn, VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse
import os
import pickle
import random

import numpy as np
from gymnasium import spaces
import common_args
from stable_baselines3 import PPO
from loguru import logger
from procgen import ProcgenEnv
from train_procgen_ppo_official import ProcgenRGBWrapper

from multiprocessing import Process, set_start_method
import multiprocessing
import tempfile

# 设置多进程启动方法为 'spawn'，解决 CUDA 在 fork 子进程中的问题
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # 如果已经设置过，则忽略错误

# 可用的procgen环境名称
PROCGEN_ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


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


def eval_models_multi_seed(model, env, num_steps=2000):
    """评估模型在多个环境中的平均奖励

    使用deterministic=True进行更准确的评估，并运行足够的步数以收集多个episode

    注意：向量化环境在done=True时会自动重置，所以不需要手动调用reset()
    """
    obs = env.reset()
    episode_rewards = []  # 记录每个episode的奖励

    # 跟踪每个环境的episode奖励和长度
    num_envs = env.num_envs
    episode_reward_per_env = np.zeros(num_envs)
    episode_length_per_env = np.zeros(num_envs, dtype=np.int32)

    for step in range(num_steps):
        # 使用deterministic=True进行更准确的评估
        action, _ = model.predict(obs, deterministic=True)
        new_obs, reward, done, info = env.step(action)

        # 累加奖励和episode长度
        episode_reward_per_env += reward
        episode_length_per_env += 1

        # 当episode结束时，记录奖励和长度并重置
        # 注意：向量化环境在done=True时已经自动reset了，new_obs是重置后的观察
        for env_idx in range(num_envs):
            if done[env_idx]:
                episode_rewards.append(episode_reward_per_env[env_idx])

                # 重置该环境的累加器
                episode_reward_per_env[env_idx] = 0
                episode_length_per_env[env_idx] = 0

        obs = new_obs  # new_obs已经是重置后的观察（如果done的话）

    num_eps = len(episode_rewards)
    logger.info(f"Completed {num_eps} episodes in {num_steps} steps")
    if num_eps == 0:
        logger.warning("No episodes completed during evaluation!")
        return 0
    else:
        mean_reward = np.mean(episode_rewards)
        logger.info(f"Mean episode reward: {mean_reward:.4f}, Std: {np.std(episode_rewards):.4f}")
        logger.info(f"Total reward: {np.sum(episode_rewards):.4f}")
        return mean_reward

def print_model_episode_stats(episode_stats, model_idx, checkpoint_version, num_models):
    """打印单个模型的episode统计信息"""
    # 筛选出在当前模型执行期间完成的episode（end_model_idx == model_idx）
    # 如果没有end_model_idx字段，则使用model_idx（向后兼容）
    model_episodes = [stat for stat in episode_stats
                     if stat.get('end_model_idx', stat.get('model_idx', -1)) == model_idx]

    if len(model_episodes) == 0:
        logger.info(f"Model {model_idx+1}/{num_models} (checkpoint {checkpoint_version}): No episodes completed during this model's execution")
        return

    rewards = [e['total_reward'] for e in model_episodes]
    lengths = [e['episode_length'] for e in model_episodes]

    logger.info(f"\n{'='*80}")
    logger.info(f"Model {model_idx+1}/{num_models} (checkpoint {checkpoint_version}) - Episodes Completed During Execution:")
    logger.info(f"{'='*80}")
    logger.info(f"  Completed Episodes: {len(model_episodes)}")
    logger.info(f"  Mean Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    logger.info(f"  Reward Range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
    logger.info(f"  Mean Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    logger.info(f"  Length Range: [{np.min(lengths)}, {np.max(lengths)}]")

    # 显示前几个episode的详细信息
    if len(model_episodes) <= 5:
        logger.info(f"  Episode Details:")
        for i, stat in enumerate(model_episodes):
            start_model = stat.get('start_model_idx', stat.get('model_idx', -1))
            logger.info(f"    Episode {i+1}: Env {stat['env_idx']}, Started in Model {start_model+1}, Reward {stat['total_reward']:.4f}, Length {stat['episode_length']}")
    else:
        logger.info(f"  First 3 episodes:")
        for i, stat in enumerate(model_episodes[:3]):
            start_model = stat.get('start_model_idx', stat.get('model_idx', -1))
            logger.info(f"    Episode {i+1}: Env {stat['env_idx']}, Started in Model {start_model+1}, Reward {stat['total_reward']:.4f}, Length {stat['episode_length']}")
        logger.info(f"  Last 3 episodes:")
        for i, stat in enumerate(model_episodes[-3:]):
            start_model = stat.get('start_model_idx', stat.get('model_idx', -1))
            logger.info(f"    Episode {len(model_episodes)-2+i}: Env {stat['env_idx']}, Started in Model {start_model+1}, Reward {stat['total_reward']:.4f}, Length {stat['episode_length']}")
    logger.info(f"{'='*80}\n")


def sample_traj_multi_seed(model_idxs, ckpt_version, env, num_envs, ckpt_path, eval_results=None, device='cuda:0', filter_data=False, min_correlation=0.2, min_increasing_ratio=0.4):
    """从多个模型checkpoint中收集轨迹数据

    新方式：40个模型，每个模型采样50步，形成2000步的数据
    最终形状：(num_envs, 2000, C, H, W)

    Args:
        eval_results: 如果提供，将使用对应模型的评估结果作为values添加到数据中
        device: 设备（cuda:0, cuda:1, cpu等）
    """
    obs = env.reset()
    episode_obs = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    episode_values = []  # 用于存储values（当relabel_reward时）
    model_num = len(model_idxs)
    length_per_model = 50
    num_models = 40  # 固定使用40个模型

    # 初始化env_trends，用于后续的数据质量过滤
    env_trends = {}

    # Episode跟踪：记录每个环境的episode信息
    # 对于每个环境，记录：episode_id, 累计奖励, episode长度, 使用的模型索引
    episode_stats = []  # 列表，每个元素是一个字典，记录一个episode的统计信息

    # 初始化每个环境的episode跟踪
    env_episode_rewards = [0.0] * num_envs  # 当前episode的累计奖励
    env_episode_lengths = [0] * num_envs  # 当前episode的长度
    env_episode_ids = [0] * num_envs  # 每个环境的episode计数器
    env_current_model_idx = [-1] * num_envs  # 当前episode使用的模型索引

    # 确保model_num至少为40
    if model_num < num_models:
        logger.warning(f"Only {model_num} models available, but need {num_models}. Using all available models.")
        num_models = model_num

    # 从model_idxs中选择40个模型（均匀采样）
    if model_num > num_models:
        selected_indices = np.linspace(0, model_num - 1, num_models, dtype=int)
        selected_model_idxs = [model_idxs[i] for i in selected_indices]
    else:
        selected_model_idxs = model_idxs[:num_models]

    logger.info(f"Using {len(selected_model_idxs)} models to collect {num_models * length_per_model} steps of data")
    logger.info(f"Using deterministic=False for action sampling (stochastic policy)")

    for model_idx, model_id in enumerate(selected_model_idxs):
        # 注意：official版本的checkpoint文件名是 checkpoint_XXX_steps.zip
        version = ckpt_version[model_id]
        model_path = f"{ckpt_path}/checkpoint_{version}_steps.zip"
        if not os.path.exists(model_path):
            # 如果checkpoint格式不存在，尝试ppo格式
            model_path = f"{ckpt_path}/ppo_{version}_steps.zip"
        if not os.path.exists(model_path):
            logger.error(f"Checkpoint file not found: {model_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        # 如果提供了eval_results，获取当前模型对应的评估结果
        if eval_results is not None:
            result = eval_results[model_id]

        model = PPO.load(model_path, env=env, device=device)
        logger.info(f"Load model {model_idx+1}/{num_models} (checkpoint {version}) from {model_path} on device {device}")

        for i in range(length_per_model):
            action, _ = model.predict(obs, deterministic=False)
            new_obs, reward, done, info = env.step(action)

            # 更新episode统计信息
            for env_idx in range(num_envs):
                # 如果这是新episode的开始（上一个step done了），记录上一个episode的统计
                # 注意：done[env_idx]表示当前step是这个episode的最后一步
                if done[env_idx] and env_episode_lengths[env_idx] > 0:
                    # 先累计当前step的reward（这是这个episode的最后一步）
                    env_episode_rewards[env_idx] += float(reward[env_idx])
                    env_episode_lengths[env_idx] += 1

                    # 记录完成的episode统计信息
                    current_checkpoint = ckpt_version[selected_model_idxs[env_current_model_idx[env_idx]]] if env_current_model_idx[env_idx] >= 0 else -1
                    episode_stats.append({
                        'env_idx': env_idx,
                        'episode_id': env_episode_ids[env_idx],
                        'model_idx': env_current_model_idx[env_idx],  # episode开始的模型（保持向后兼容）
                        'start_model_idx': env_current_model_idx[env_idx],  # episode开始的模型
                        'end_model_idx': model_idx,  # episode结束的模型（当前模型）
                        'model_checkpoint': current_checkpoint,
                        'total_reward': env_episode_rewards[env_idx],
                        'episode_length': env_episode_lengths[env_idx],
                        'step_in_trajectory': model_idx * length_per_model + i - env_episode_lengths[env_idx] + 1
                    })

                    # 重置episode统计，准备下一个episode
                    env_episode_rewards[env_idx] = 0.0
                    env_episode_lengths[env_idx] = 0
                    env_episode_ids[env_idx] += 1
                    env_current_model_idx[env_idx] = model_idx  # 新episode使用当前模型
                elif done[env_idx]:
                    # 如果done=True但length=0，说明这是新episode的第一步
                    env_current_model_idx[env_idx] = model_idx
                    env_episode_rewards[env_idx] = float(reward[env_idx])
                    env_episode_lengths[env_idx] = 1
                else:
                    # 继续当前episode
                    if env_current_model_idx[env_idx] < 0:
                        # 如果还没有初始化，初始化当前episode
                        env_current_model_idx[env_idx] = model_idx
                    env_episode_rewards[env_idx] += float(reward[env_idx])
                    env_episode_lengths[env_idx] += 1

            # 记录轨迹数据
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            # 如果提供了eval_results，添加values
            if eval_results is not None:
                episode_values.append(np.ones(reward.shape) * result)
            episode_dones.append(done)
            obs = new_obs

        # 显式删除模型并清理显存
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 在每个模型执行完后，立即打印该模型的episode统计信息
        print_model_episode_stats(episode_stats, model_idx, version, num_models)

        # 在模型切换时，检查是否有未完成的episode需要记录
        # 注意：我们不在这里强制结束episode，让它们自然结束
        # 但如果episode跨越了模型边界，它仍然使用开始时的模型索引

    # 处理最后未完成的episodes（在数据收集结束时）
    for env_idx in range(num_envs):
        if env_episode_lengths[env_idx] > 0:
            current_checkpoint = ckpt_version[selected_model_idxs[env_current_model_idx[env_idx]]] if env_current_model_idx[env_idx] >= 0 else -1
            episode_stats.append({
                'env_idx': env_idx,
                'episode_id': env_episode_ids[env_idx],
                'model_idx': env_current_model_idx[env_idx],  # episode开始的模型（保持向后兼容）
                'start_model_idx': env_current_model_idx[env_idx],  # episode开始的模型
                'end_model_idx': num_models - 1,  # 在最后一个模型执行期间结束（虽然未完成）
                'model_checkpoint': current_checkpoint,
                'total_reward': env_episode_rewards[env_idx],
                'episode_length': env_episode_lengths[env_idx],
                'step_in_trajectory': num_models * length_per_model - env_episode_lengths[env_idx],
                'note': 'incomplete_episode'
            })

    # 输出episode统计信息
    if len(episode_stats) > 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"Episode Statistics Summary (Total {len(episode_stats)} episodes)")
        logger.info(f"{'='*80}")

        # 按模型索引分组统计
        model_episodes = {}
        for stat in episode_stats:
            model_idx = stat['model_idx']
            if model_idx not in model_episodes:
                model_episodes[model_idx] = []
            model_episodes[model_idx].append(stat)

        # 输出每个模型的episode统计
        for model_idx in sorted(model_episodes.keys()):
            episodes = model_episodes[model_idx]
            rewards = [e['total_reward'] for e in episodes]
            lengths = [e['episode_length'] for e in episodes]
            checkpoint = episodes[0]['model_checkpoint']

            logger.info(f"\nModel {model_idx+1}/{num_models} (checkpoint {checkpoint}):")
            logger.info(f"  Episodes: {len(episodes)}")
            logger.info(f"  Mean Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
            logger.info(f"  Reward Range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
            logger.info(f"  Mean Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
            logger.info(f"  Length Range: [{np.min(lengths)}, {np.max(lengths)}]")

        # 分析是否越往后表现越好（按模型索引）
        logger.info(f"\n{'='*80}")
        logger.info("Performance Trend Analysis (by model index):")
        logger.info(f"{'='*80}")

        # 计算每个模型索引的平均奖励
        model_avg_rewards = {}
        for model_idx in sorted(model_episodes.keys()):
            episodes = model_episodes[model_idx]
            rewards = [e['total_reward'] for e in episodes]
            model_avg_rewards[model_idx] = np.mean(rewards)

        # 输出趋势
        model_indices = sorted(model_avg_rewards.keys())
        logger.info(f"Model indices: {model_indices}")
        logger.info(f"Average rewards: {[model_avg_rewards[i] for i in model_indices]}")

        # 计算趋势（简单线性回归）
        if len(model_indices) > 1:
            x = np.array(model_indices)
            y = np.array([model_avg_rewards[i] for i in model_indices])
            # 简单线性回归
            coeff = np.polyfit(x, y, 1)
            slope = coeff[0]
            logger.info(f"Linear trend slope: {slope:.6f}")
            if slope > 0:
                logger.info(f"  → Performance is IMPROVING as model index increases (later models perform better)")
            elif slope < 0:
                logger.info(f"  → Performance is DECREASING as model index increases (later models perform worse)")
            else:
                logger.info(f"  → Performance is STABLE across models")

        # 分析每个环境的reward趋势（随着episode增加，reward是否递增）
        logger.info(f"\n{'='*80}")
        logger.info("Reward Trend Analysis (by environment, as episode increases):")
        logger.info(f"{'='*80}")

        # 按环境分组episode
        env_episodes = {}
        for stat in episode_stats:
            env_idx = stat['env_idx']
            if env_idx not in env_episodes:
                env_episodes[env_idx] = []
            env_episodes[env_idx].append(stat)

        # 对每个环境，按episode_id排序并分析趋势（不输出详细信息，只收集数据）
        env_trends = {}
        for env_idx in sorted(env_episodes.keys()):
            episodes = env_episodes[env_idx]
            # 按episode_id排序
            episodes_sorted = sorted(episodes, key=lambda x: x['episode_id'])

            if len(episodes_sorted) < 2:
                continue  # 跳过只有一个episode的环境

            # 提取reward序列
            rewards = [e['total_reward'] for e in episodes_sorted]
            episode_ids = [e['episode_id'] for e in episodes_sorted]

            # 计算线性回归斜率
            x = np.array(episode_ids)
            y = np.array(rewards)
            coeff = np.polyfit(x, y, 1)
            slope = coeff[0]
            intercept = coeff[1]

            # 计算相关系数（Pearson correlation）
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0

            # 计算单调递增的比例（相邻episode中reward增加的次数）
            increasing_count = sum(1 for i in range(len(rewards)-1) if rewards[i+1] > rewards[i])
            increasing_ratio = increasing_count / (len(rewards) - 1) if len(rewards) > 1 else 0

            env_trends[env_idx] = {
                'slope': slope,
                'correlation': correlation,
                'increasing_ratio': increasing_ratio,
                'num_episodes': len(episodes_sorted),
                'first_reward': rewards[0],
                'last_reward': rewards[-1],
                'mean_reward': np.mean(rewards),
                'rewards': rewards
            }

        # 汇总所有环境的趋势
        if len(env_trends) > 0:
            logger.info(f"Analyzed {len(env_trends)} environments")

            all_slopes = [env_trends[env_idx]['slope'] for env_idx in sorted(env_trends.keys())]
            all_correlations = [env_trends[env_idx]['correlation'] for env_idx in sorted(env_trends.keys())]
            all_increasing_ratios = [env_trends[env_idx]['increasing_ratio'] for env_idx in sorted(env_trends.keys())]

            positive_slope_count = sum(1 for s in all_slopes if s > 0)
            positive_corr_count = sum(1 for c in all_correlations if c > 0.3)
            high_increasing_ratio_count = sum(1 for r in all_increasing_ratios if r > 0.5)

            logger.info(f"\n{'='*80}")
            logger.info("Summary across all environments:")
            logger.info(f"{'='*80}")
            logger.info(f"Environments with positive slope: {positive_slope_count}/{len(env_trends)} ({positive_slope_count/len(env_trends)*100:.1f}%)")
            logger.info(f"Environments with strong positive correlation (>0.3): {positive_corr_count}/{len(env_trends)} ({positive_corr_count/len(env_trends)*100:.1f}%)")
            logger.info(f"Environments with >50% increasing transitions: {high_increasing_ratio_count}/{len(env_trends)} ({high_increasing_ratio_count/len(env_trends)*100:.1f}%)")
            logger.info(f"Mean slope across all environments: {np.mean(all_slopes):.6f} ± {np.std(all_slopes):.6f}")
            logger.info(f"Mean correlation across all environments: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}")
            logger.info(f"Mean increasing ratio across all environments: {np.mean(all_increasing_ratios):.2%} ± {np.std(all_increasing_ratios):.2%}")

            # 输出几个代表性环境的详细信息（趋势最明显的几个）
            logger.info(f"\n{'='*80}")
            logger.info("Sample environments (showing most representative trends):")
            logger.info(f"{'='*80}")

            # 找出趋势最明显的几个环境
            # 1. 斜率最大（最递增）的3个
            # 2. 斜率最小（最递减）的3个
            # 3. 相关系数最大的3个（趋势最稳定）

            sorted_by_slope = sorted(env_trends.items(), key=lambda x: x[1]['slope'], reverse=True)
            sorted_by_corr = sorted(env_trends.items(), key=lambda x: x[1]['correlation'], reverse=True)

            sample_envs = set()
            # 取前3个和后3个斜率最大的
            for env_idx, _ in sorted_by_slope[:3]:
                sample_envs.add(env_idx)
            for env_idx, _ in sorted_by_slope[-3:]:
                sample_envs.add(env_idx)
            # 取前3个相关系数最大的
            for env_idx, _ in sorted_by_corr[:3]:
                sample_envs.add(env_idx)

            # 输出这些代表性环境的详细信息
            for env_idx in sorted(sample_envs):
                trend = env_trends[env_idx]
                rewards = trend['rewards']

                # 判断趋势
                slope = trend['slope']
                correlation = trend['correlation']
                if slope > 0 and correlation > 0.3:
                    trend_desc = "STRONGLY INCREASING"
                elif slope > 0:
                    trend_desc = "SLIGHTLY INCREASING"
                elif slope < 0 and correlation < -0.3:
                    trend_desc = "STRONGLY DECREASING"
                elif slope < 0:
                    trend_desc = "SLIGHTLY DECREASING"
                else:
                    trend_desc = "STABLE"

                increasing_count = sum(1 for i in range(len(rewards)-1) if rewards[i+1] > rewards[i])

                logger.info(f"Env {env_idx}: {trend['num_episodes']} episodes")
                logger.info(f"  Linear slope: {slope:.6f}, Correlation: {correlation:.4f}")
                logger.info(f"  Increasing ratio: {trend['increasing_ratio']:.2%} ({increasing_count}/{len(rewards)-1} transitions)")
                logger.info(f"  First episode reward: {trend['first_reward']:.4f}, Last episode reward: {trend['last_reward']:.4f}")
                logger.info(f"  Mean reward: {trend['mean_reward']:.4f} ± {np.std(rewards):.4f}")
                logger.info(f"  Trend: {trend_desc}")

        # 输出前10个和后10个episode的详细信息（用于观察趋势）
        logger.info(f"\n{'='*80}")
        logger.info("First 10 episodes:")
        logger.info(f"{'='*80}")
        for i, stat in enumerate(episode_stats[:10]):
            logger.info(f"Episode {i+1}: Env {stat['env_idx']}, Model {stat['model_idx']+1}, "
                       f"Reward {stat['total_reward']:.4f}, Length {stat['episode_length']}")

        if len(episode_stats) > 10:
            logger.info(f"\nLast 10 episodes:")
            logger.info(f"{'='*80}")
            for i, stat in enumerate(episode_stats[-10:]):
                logger.info(f"Episode {len(episode_stats)-9+i}: Env {stat['env_idx']}, Model {stat['model_idx']+1}, "
                           f"Reward {stat['total_reward']:.4f}, Length {stat['episode_length']}")

        logger.info(f"\n{'='*80}\n")
    else:
        logger.warning("No episodes completed during data collection!")

    # 整理轨迹数据
    # obs shape: (num_envs, C, H, W) from ProcgenRGBWrapper
    if len(episode_obs) == 0:
        logger.error("No observations collected!")
        raise ValueError("No observations collected in sample_traj_multi_seed")

    total_steps = num_models * length_per_model  # 应该是2000
    logger.info(f"Collected {len(episode_obs)} steps of data (expected {total_steps})")

    if eval_results is None:
        trajectory = {
            "observations": np.stack(episode_obs),  # (total_steps, num_envs, C, H, W)
            "actions": np.array(episode_actions),  # (total_steps, num_envs)
            "rewards": np.array(episode_rewards),  # (total_steps, num_envs)
            "dones": np.array(episode_dones),  # (total_steps, num_envs)
        }
    else:
        trajectory = {
            "observations": np.stack(episode_obs),  # (total_steps, num_envs, C, H, W)
            "actions": np.array(episode_actions),  # (total_steps, num_envs)
            "rewards": np.array(episode_rewards),  # (total_steps, num_envs)
            "dones": np.array(episode_dones),  # (total_steps, num_envs)
            "values": np.array(episode_values),  # (total_steps, num_envs)
        }

    logger.info(f"Before reshape - obs: {trajectory['observations'].shape}, actions: {trajectory['actions'].shape}")

    # 重塑为 (num_envs, total_steps, C, H, W)
    # 从 (total_steps, num_envs, C, H, W) -> (num_envs, total_steps, C, H, W)
    obs_shape = trajectory['observations'].shape[2:]  # (C, H, W)
    actual_steps = trajectory['observations'].shape[0]

    if actual_steps != total_steps:
        logger.warning(f"Total steps mismatch: got {actual_steps}, expected {total_steps}")

    try:
        # transpose: (total_steps, num_envs, C, H, W) -> (num_envs, total_steps, C, H, W)
        trajectory['observations'] = trajectory['observations'].transpose((1, 0, 2, 3, 4))
        trajectory['actions'] = trajectory['actions'].transpose((1, 0))  # (num_envs, total_steps)
        trajectory['rewards'] = trajectory['rewards'].transpose((1, 0))  # (num_envs, total_steps)
        trajectory['dones'] = trajectory['dones'].transpose((1, 0))  # (num_envs, total_steps)
        # 如果包含values，也需要transpose
        if 'values' in trajectory:
            trajectory['values'] = trajectory['values'].transpose((1, 0))  # (num_envs, total_steps)
    except Exception as e:
        logger.error(f"Error during transpose: {str(e)}")
        logger.error(f"Shapes: obs={trajectory['observations'].shape}, actions={trajectory['actions'].shape}")
        raise

    logger.info(f"After reshape - obs: {trajectory['observations'].shape}, act: {trajectory['actions'].shape}")

    # 优化数据类型以节省内存和存储空间
    # 保持observations为uint8格式（1字节/像素），不转换为float16（2字节/像素）
    # 归一化可以在训练时动态进行，不需要在存储时归一化
    if trajectory['observations'].dtype != np.uint8:
        # 如果已经是uint8，保持；否则转换为uint8（假设值域在0-255）
        if trajectory['observations'].max() <= 255 and trajectory['observations'].min() >= 0:
            trajectory['observations'] = trajectory['observations'].astype(np.uint8)
            logger.info(f"Converted observations to uint8 format")
        else:
            logger.warning(f"Observations value range [{trajectory['observations'].min()}, {trajectory['observations'].max()}] not in [0, 255], keeping original dtype")

    # 确保actions和rewards使用最小数据类型
    if trajectory['actions'].dtype != np.int32 and trajectory['actions'].dtype != np.int64:
        trajectory['actions'] = trajectory['actions'].astype(np.int32)

    if trajectory['rewards'].dtype != np.float32 and trajectory['rewards'].dtype != np.float16:
        trajectory['rewards'] = trajectory['rewards'].astype(np.float32)

    if trajectory['dones'].dtype != bool:
        trajectory['dones'] = trajectory['dones'].astype(bool)

    # 确保values使用float32类型
    if 'values' in trajectory:
        if trajectory['values'].dtype != np.float32 and trajectory['values'].dtype != np.float16:
            trajectory['values'] = trajectory['values'].astype(np.float32)
        logger.info(f"Final data types - obs: {trajectory['observations'].dtype}, actions: {trajectory['actions'].dtype}, rewards: {trajectory['rewards'].dtype}, dones: {trajectory['dones'].dtype}, values: {trajectory['values'].dtype}")
    else:
        logger.info(f"Final data types - obs: {trajectory['observations'].dtype}, actions: {trajectory['actions'].dtype}, rewards: {trajectory['rewards'].dtype}, dones: {trajectory['dones'].dtype}")

    # 数据质量过滤：根据环境趋势过滤掉不符合理想趋势的环境
    if filter_data and len(env_trends) > 0:
        # 设置过滤阈值
        min_slope = 0.0  # 最小斜率（>=0表示正趋势）

        # 筛选符合条件的环境索引
        valid_env_indices = []
        filtered_env_indices = []

        for env_idx in range(num_envs):
            if env_idx in env_trends:
                trend = env_trends[env_idx]
                slope = trend['slope']
                correlation = trend['correlation']
                increasing_ratio = trend['increasing_ratio']

                # 判断是否符合过滤条件（同时满足两个条件）
                if (slope >= min_slope and
                    correlation >= min_correlation and
                    increasing_ratio >= min_increasing_ratio):
                    valid_env_indices.append(env_idx)
                else:
                    filtered_env_indices.append(env_idx)
            else:
                # 如果环境没有足够的episode进行分析，默认保留
                # 但也可以选择过滤掉，这里选择保留
                valid_env_indices.append(env_idx)

        logger.info(f"\n{'='*80}")
        logger.info("Data Quality Filtering:")
        logger.info(f"{'='*80}")
        logger.info(f"Filter criteria:")
        logger.info(f"  Min slope: {min_slope}")
        logger.info(f"  Min correlation: {min_correlation}")
        logger.info(f"  Min increasing ratio: {min_increasing_ratio}")
        logger.info(f"Valid environments: {len(valid_env_indices)}/{num_envs} ({len(valid_env_indices)/num_envs*100:.1f}%)")
        logger.info(f"Filtered environments: {len(filtered_env_indices)}/{num_envs} ({len(filtered_env_indices)/num_envs*100:.1f}%)")

        if len(valid_env_indices) == 0:
            logger.error("All environments were filtered out! Please adjust filter criteria.")
            raise ValueError("No valid environments after filtering")

        if len(valid_env_indices) < num_envs:
            # 只保留符合条件的环境
            valid_env_indices = np.array(valid_env_indices)
            trajectory['observations'] = trajectory['observations'][valid_env_indices]
            trajectory['actions'] = trajectory['actions'][valid_env_indices]
            trajectory['rewards'] = trajectory['rewards'][valid_env_indices]
            trajectory['dones'] = trajectory['dones'][valid_env_indices]
            if 'values' in trajectory:
                trajectory['values'] = trajectory['values'][valid_env_indices]

            logger.info(f"After filtering - obs: {trajectory['observations'].shape}, act: {trajectory['actions'].shape}")
            if len(filtered_env_indices) <= 20:
                logger.info(f"Filtered out environments: {filtered_env_indices}")
            else:
                logger.info(f"Filtered out {len(filtered_env_indices)} environments: {filtered_env_indices[:20]}... (showing first 20)")
        else:
            logger.info("All environments passed the filter criteria.")
    elif filter_data:
        logger.warning("Filtering requested but no environment trends available. Skipping filter.")

    return trajectory

def make_procgen_env(num_envs, env_name, num_levels=0, start_level=0, distribution_mode="hard", rand_seed=None, seed_list=None):
    """创建procgen向量化环境

    如果提供了有效的seed_list（所有值都非-1），则为每个环境创建独立的环境，
    每个环境使用seed_list[i]作为start_level，num_levels=1，确保每个环境固定在一个关卡上。
    否则，创建单个向量化环境，所有环境共享相同的start_level和num_levels。
    """
    if seed_list is None:
        seed_list = [-1] * num_envs

    # 如果提供了有效的seed_list（所有值都非-1），为每个环境创建独立的环境，每个环境固定在一个关卡上
    if all(s != -1 for s in seed_list):
        assert num_envs == len(seed_list), "Number of environments must match number of seeds"
        # 为每个环境创建独立的环境，每个环境固定在一个关卡上
        envs = []
        for i in range(num_envs):
            env = ProcgenEnv(
                num_envs=1,
                env_name=env_name,
                num_levels=1,  # 每个环境只有1个关卡
                start_level=seed_list[i],  # 第i个环境使用关卡 seed_list[i]
                distribution_mode=distribution_mode,
                rand_seed=rand_seed,
            )
            # 提取RGB观察并转换为channel-first格式
            env = ProcgenRGBWrapper(env, "rgb")
            envs.append(env)

        # 使用自定义的向量化环境合并多个环境
        env = MultiProcgenVecEnv(envs)
        return env

    # 否则，创建单个向量化环境（原有逻辑）
    if num_envs == 1:
        env = ProcgenEnv(
            num_envs=1,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            rand_seed=rand_seed,
        )
        env = ProcgenRGBWrapper(env, "rgb")
        return env
    else:
        # 创建单个向量化环境包含多个子环境
        env = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            rand_seed=rand_seed,
        )
        env = ProcgenRGBWrapper(env, "rgb")
        return env

def process_slice(slice_idx, slices, int_samples, checkpoint_version, temp_dir, ckpt_path, num_per_slice,
                  eval_results=None, env_name="coinrun", num_levels=0, start_level=0, distribution_mode="hard", device='cuda:0', seed_list=None,
                  filter_data=False, min_correlation=0.2, min_increasing_ratio=0.4):
    """处理单个 slice 的数据收集任务，并将结果保存到临时文件"""
    sample_env = None
    try:
        logger.info(f"Slice {slice_idx+1}/{slices} started")
        # 如果提供了seed_list，为当前slice创建对应的seed_list
        slice_seed_list = None
        if seed_list is not None:
            slice_start_idx = slice_idx * num_per_slice
            slice_seed_list = seed_list[slice_start_idx:slice_start_idx + num_per_slice]

        sample_env = make_procgen_env(
            num_envs=num_per_slice,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level + slice_idx * num_per_slice,
            distribution_mode=distribution_mode,
            seed_list=slice_seed_list,
        )
        logger.info(f"Slice {slice_idx+1}/{slices}: Environment created, starting data collection...")
        sample_res = sample_traj_multi_seed(
            int_samples, checkpoint_version, sample_env, num_envs=num_per_slice, ckpt_path=ckpt_path, eval_results=eval_results, device=device,
            filter_data=filter_data, min_correlation=min_correlation, min_increasing_ratio=min_increasing_ratio
        )

        # 验证数据
        if sample_res is None:
            raise ValueError("sample_traj_multi_seed returned None")
        if 'observations' not in sample_res:
            raise ValueError("sample_res missing 'observations' key")

        logger.info(f"Slice {slice_idx+1}/{slices}: Data collection completed, saving to file...")
        logger.info(f"Slice {slice_idx+1}/{slices}: Data shapes - obs: {sample_res['observations'].shape}, actions: {sample_res['actions'].shape}")

        # 将结果保存到临时文件（使用不压缩格式以加快保存速度）
        temp_file = os.path.join(temp_dir, f"slice_{slice_idx}.npz")
        logger.info(f"Slice {slice_idx+1}/{slices}: Saving to {temp_file}...")
        # 使用np.savez（不压缩）以加快保存速度
        if 'values' in sample_res:
            np.savez(
                temp_file,
                observations=sample_res['observations'],
                actions=sample_res['actions'],
                rewards=sample_res['rewards'],
                dones=sample_res['dones'],
                values=sample_res['values']
            )
        else:
            np.savez(
                temp_file,
                observations=sample_res['observations'],
                actions=sample_res['actions'],
                rewards=sample_res['rewards'],
                dones=sample_res['dones']
            )

        # 验证文件是否成功保存
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"File {temp_file} was not created after saving!")
        file_size = os.path.getsize(temp_file)
        logger.info(f"Slice {slice_idx+1}/{slices}: File saved successfully, size: {file_size} bytes")

        logger.info(f"Slice {slice_idx+1}/{slices} completed and saved to {temp_file}")
        if sample_env is not None:
            sample_env.close()
        return True
    except Exception as e:
        logger.error(f"Error in slice {slice_idx+1}/{slices}: {str(e)}")
        import traceback
        logger.error(f"Traceback for slice {slice_idx+1}/{slices}:")
        traceback.print_exc()
        # 确保环境被关闭
        if sample_env is not None:
            try:
                sample_env.close()
            except:
                pass
        return False

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    parser.add_argument("--fix_seed", action="store_true",
                        default=False, help="Fix seed or not")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--cnn", action="store_false",
                        default=True, help="Use CnnPolicy or not")
    parser.add_argument("--train", action="store_true",
                        default=False, help="Collect train data or not")
    parser.add_argument("--num_levels", type=int, default=0,
                        help="Number of unique levels (0 for unlimited)")
    parser.add_argument("--start_level", type=int, default=0,
                        help="Start level seed")
    parser.add_argument("--distribution_mode", type=str, default="hard",
                        choices=["easy", "hard", "extreme", "memory", "exploration"],
                        help="Distribution mode for procgen")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Checkpoint save path (default: models/Procgen/{env}/official)")
    parser.add_argument("--relabel_reward", action="store_true",
                        default=False, help="Relabel reward using model evaluation results as values")
    parser.add_argument("--device", type=str, default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help='Device to use (cuda:0, cuda:1, or cpu)')
    parser.add_argument("--filter_data", action="store_true", default=False,
                        help="Filter out environments with poor trend quality")
    parser.add_argument("--min_correlation", type=float, default=0.2,
                        help="Minimum correlation coefficient for filtering (default: 0.2)")
    parser.add_argument("--min_increasing_ratio", type=float, default=0.4,
                        help="Minimum increasing ratio for filtering (default: 0.4)")
    parser.add_argument("--data-dir", default="datasets/Procgen")
    args = vars(parser.parse_args())
    print("Args: ", args)

    env_name = args['env']
    if env_name not in PROCGEN_ENV_NAMES: parser.error('Unknown Procgen environment.')
    n_envs = args['envs']
    num_levels = args['num_levels']
    start_level = args['start_level']
    distribution_mode = args['distribution_mode']

    seed = args['seed']
    relabel_reward = args['relabel_reward']
    device = args['device']
    filter_data = args['filter_data']
    min_correlation = args['min_correlation']
    min_increasing_ratio = args['min_increasing_ratio']

    # 设置CUDA设备（如果使用GPU）
    if device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            logger.warning(f"CUDA not available, falling back to CPU")
            device = 'cpu'
        else:
            # 提取GPU编号
            gpu_id = device.split(':')[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            device = 'cuda:0'  # 设置后，GPU会被重新编号为cuda:0
            logger.info(f"Using GPU {gpu_id} (visible as cuda:0)")
    else:
        logger.info(f"Using CPU")

    assert args['cnn'], "Only support cnn models"

    if not args['fix_seed']:
        logger.info(f"no fix seed setting")
        num_envs = 20
        env = make_procgen_env(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
        )
    else:
        logger.info(f"set seed to {seed}")
        env = make_procgen_env(
            num_envs=1,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            rand_seed=seed,
        )

    if not args['fix_seed']:
        # Step1: eval_checkpoints_and_get_avg_return
        # 使用save_path参数或默认路径
        if args['save_path']:
            checkpoint_path = args['save_path']
        else:
            checkpoint_path = f"models/Procgen/{env_name}/official"

        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint path {checkpoint_path} does not exist!")
            raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist!")

        checkpoint_list = os.listdir(checkpoint_path)
        # 支持两种checkpoint命名格式：checkpoint_XXX_steps.zip 和 ppo_XXX_steps.zip
        checkpoint_version = []
        for ckpt in checkpoint_list:
            if ckpt.endswith('.zip') and ckpt not in ["final_model.zip", "final.zip"]:
                if ckpt.startswith('checkpoint_'):
                    # checkpoint_XXX_steps.zip
                    version_str = ckpt.split('_')[1]
                elif ckpt.startswith('ppo_'):
                    # ppo_XXX_steps.zip
                    version_str = ckpt.split('_')[1]
                else:
                    continue
                try:
                    checkpoint_version.append(int(version_str))
                except ValueError:
                    continue

        checkpoint_version.sort()

        if len(checkpoint_version) == 0:
            logger.error(f"No valid checkpoints found in {checkpoint_path}!")
            raise ValueError(f"No valid checkpoints found in {checkpoint_path}!")

        logger.info(f"Found {len(checkpoint_version)} checkpoints")

        # 注释掉模型评估部分，因为不再需要根据模型表现选择samples
        # eval_results = []
        # valid_checkpoint_version = []
        # for v in checkpoint_version:
        #     # 尝试两种格式
        #     model_path = f"{checkpoint_path}/checkpoint_{v}_steps.zip"
        #     if not os.path.exists(model_path):
        #         model_path = f"{checkpoint_path}/ppo_{v}_steps.zip"
        #     if not os.path.exists(model_path):
        #         logger.warning(f"Checkpoint {v} not found, skipping")
        #         continue
        #     model = PPO.load(model_path, env=env)
        #     valid_checkpoint_version.append(v)
        #     avg_reward = eval_models_multi_seed(model, env)
        #     logger.info(f"Avg reward {avg_reward} at {v} steps")
        #     eval_results.append(avg_reward)
        #
        # # 更新checkpoint_version为有效的版本号列表
        # checkpoint_version = valid_checkpoint_version
        #
        # if len(eval_results) == 0:
        #     logger.error("No valid checkpoints found for evaluation!")
        #     raise ValueError("No valid checkpoints found for evaluation!")
        #
        # with open(f"{checkpoint_path}/eval_results.txt", "w") as f:
        #     eval_results_str = [str(v) for v in eval_results]
        #     f.write(", ".join(eval_results_str))

        # 验证checkpoint文件是否存在
        valid_checkpoint_version = []
        for v in checkpoint_version:
            model_path = f"{checkpoint_path}/checkpoint_{v}_steps.zip"
            if not os.path.exists(model_path):
                model_path = f"{checkpoint_path}/ppo_{v}_steps.zip"
            if os.path.exists(model_path):
                valid_checkpoint_version.append(v)
            else:
                logger.warning(f"Checkpoint {v} not found, skipping")

        checkpoint_version = valid_checkpoint_version
        if len(checkpoint_version) == 0:
            logger.error("No valid checkpoints found!")
            raise ValueError("No valid checkpoints found!")

        # 如果启用relabel_reward，评估所有checkpoint并获取eval_results
        eval_results = None
        if relabel_reward:
            eval_results_file = os.path.join(checkpoint_path, "eval_results.txt")
            if os.path.exists(eval_results_file):
                # 从文件加载评估结果
                with open(eval_results_file, "r") as f:
                    eval_results_str = f.read()
                eval_results = eval_results_str.split(", ")
                eval_results = [float(v) for v in eval_results]
                logger.info(f"Loaded eval results from {eval_results_file}")
                # 确保eval_results的长度与checkpoint_version一致
                if len(eval_results) != len(checkpoint_version):
                    logger.warning(f"eval_results length ({len(eval_results)}) != checkpoint_version length ({len(checkpoint_version)}), re-evaluating...")
                    eval_results = None

            if eval_results is None:
                # 评估所有checkpoint
                logger.info("Evaluating all checkpoints to get average returns for relabel_reward...")
                eval_results = []
                for v in checkpoint_version:
                    model_path = f"{checkpoint_path}/checkpoint_{v}_steps.zip"
                    if not os.path.exists(model_path):
                        model_path = f"{checkpoint_path}/ppo_{v}_steps.zip"
                    if not os.path.exists(model_path):
                        logger.warning(f"Checkpoint {v} not found, skipping")
                        eval_results.append(0.0)  # 使用0作为占位符
                        continue
                    model = PPO.load(model_path, env=env, device=device)
                    avg_reward = eval_models_multi_seed(model, env)
                    logger.info(f"Avg reward {avg_reward} at {v} steps")
                    eval_results.append(avg_reward)
                    del model
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 对eval_results进行归一化：使得最高值为1
                eval_results = np.array(eval_results)
                max_value = np.max(eval_results)
                if max_value > 0:
                    eval_results = eval_results / max_value
                    logger.info(f"Normalized eval_results: max_value={max_value:.4f}, normalized range=[{np.min(eval_results):.4f}, {np.max(eval_results):.4f}]")
                else:
                    logger.warning(f"All eval_results are zero or negative, skipping normalization")
                eval_results = eval_results.tolist()

                # 保存评估结果（保存归一化后的值）
                with open(eval_results_file, "w") as f:
                    eval_results_str = [str(v) for v in eval_results]
                    f.write(", ".join(eval_results_str))
                logger.info(f"Saved normalized eval results to {eval_results_file}")
            else:
                # 如果从文件加载，也需要进行归一化（以防文件中的值还没有归一化）
                eval_results = np.array(eval_results)
                max_value = np.max(eval_results)
                if max_value > 0 and max_value != 1.0:
                    # 如果最大值不是1，说明还没有归一化，进行归一化
                    eval_results = eval_results / max_value
                    logger.info(f"Normalized loaded eval_results: max_value={max_value:.4f}, normalized range=[{np.min(eval_results):.4f}, {np.max(eval_results):.4f}]")
                    # 保存归一化后的值
                    with open(eval_results_file, "w") as f:
                        eval_results_str = [str(v) for v in eval_results]
                        f.write(", ".join(eval_results_str))
                    logger.info(f"Updated eval_results file with normalized values")
                eval_results = eval_results.tolist()

        # 从所有checkpoint index中均匀选取40个samples
        num_checkpoints = len(checkpoint_version)
        num_samples = 40
        if num_checkpoints <= num_samples:
            # 如果checkpoint数量少于等于40，则全部选择
            int_samples = list(range(num_checkpoints))
        else:
            # 均匀选取40个index
            samples = np.linspace(0, num_checkpoints - 1, num_samples)
            int_samples = np.round(samples).astype(int).tolist()
            # 确保没有重复
            int_samples = sorted(list(set(int_samples)))

        logger.info(f"Collect {len(int_samples)} samples from {num_checkpoints} checkpoints: {int_samples}")

        collect_train = args["train"]
        # 设置多进程
        if collect_train:
            # 使用800个环境，每个环境2000步数据
            sample_env_num = 1200
            slices = 1
            num_per_slice = sample_env_num // slices

            # 生成seed_list：为每个环境设置固定的关卡
            # 如果 num_levels=0，则使用 start_level + i 作为每个环境的固定关卡
            # 如果 num_levels>0，则使用 start_level 到 start_level+num_levels-1 的范围
            if num_levels == 0:
                # 无限关卡模式：为每个环境设置固定关卡 start_level + i
                seed_list = [start_level + i for i in range(sample_env_num)]
            else:
                # 有限关卡模式：循环使用 start_level 到 start_level+num_levels-1
                seed_list = [start_level + (i % num_levels) for i in range(sample_env_num)]

            # 创建临时目录保存结果
            temp_dir = tempfile.mkdtemp(prefix="procgen_data_")
            logger.info(f"Created temporary directory for results: {temp_dir}")

            # 创建并启动进程
            processes = []
            for slice_idx in range(slices):
                p = Process(
                    target=process_slice,
                    args=(slice_idx, slices, int_samples, checkpoint_version, temp_dir, checkpoint_path,
                          num_per_slice, eval_results, env_name, num_levels, start_level, distribution_mode, device, seed_list,
                          filter_data, min_correlation, min_increasing_ratio)
                )
                processes.append(p)
                p.start()

            # 等待所有进程完成
            for i, p in enumerate(processes):
                p.join()
                exit_code = p.exitcode
                if exit_code != 0:
                    raise RuntimeError(f"Collector worker {i} exited with code {exit_code}")

            # 从临时文件中加载结果
            sample_res_list = []
            for slice_idx in range(slices):
                temp_file = os.path.join(temp_dir, f"slice_{slice_idx}.npz")
                logger.info(f"Checking for results file: {temp_file}")
                if os.path.exists(temp_file):
                    try:
                        file_size = os.path.getsize(temp_file)
                        logger.info(f"Found file {temp_file}, size: {file_size} bytes")
                        # 从压缩的npz文件加载
                        data = np.load(temp_file)
                        sample_res = {
                            'observations': data['observations'],
                            'actions': data['actions'],
                            'rewards': data['rewards'],
                            'dones': data['dones']
                        }
                        # 如果包含values，也加载
                        if 'values' in data:
                            sample_res['values'] = data['values']
                        data.close()  # 关闭文件以释放资源
                        sample_res_list.append(sample_res)
                        logger.info(f"Loaded results from {temp_file}")
                    except Exception as e:
                        logger.error(f"Error loading results from {temp_file}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning(f"Results for slice {slice_idx+1}/{slices} not found at {temp_file}")
                    # 列出临时目录中的所有文件，帮助调试
                    if os.path.exists(temp_dir):
                        files_in_dir = os.listdir(temp_dir)
                        logger.warning(f"Files in temp_dir {temp_dir}: {files_in_dir}")

            # 检查是否有足够的结果进行合并
            if len(sample_res_list) != slices:
                logger.error("No results collected from any process!")
                raise RuntimeError("Failed to collect any results from processes")

            # 合并所有 slice 的结果
            logger.info(f"Merging results from {len(sample_res_list)} slices")
            try:
                sample_res = {
                    "observations": np.concatenate([res['observations'] for res in sample_res_list], axis=0),
                    "actions": np.concatenate([res['actions'] for res in sample_res_list], axis=0),
                    "rewards": np.concatenate([res['rewards'] for res in sample_res_list], axis=0),
                    "dones": np.concatenate([res['dones'] for res in sample_res_list], axis=0),
                }
                # 如果包含values，也合并
                if 'values' in sample_res_list[0]:
                    sample_res['values'] = np.concatenate([res['values'] for res in sample_res_list], axis=0)
                logger.info(f"obs: {sample_res['observations'].shape}")
                logger.info(f"act: {sample_res['actions'].shape}")
                if 'values' in sample_res:
                    logger.info(f"values: {sample_res['values'].shape}")
            except Exception as e:
                logger.error(f"Error merging results: {str(e)}")
                raise
        else:
            # 生成seed_list：为每个环境设置固定的关卡
            num_test_envs = 40
            if num_levels == 0:
                # 无限关卡模式：为每个环境设置固定关卡 start_level + i
                test_seed_list = [start_level + i for i in range(num_test_envs)]
            else:
                # 有限关卡模式：循环使用 start_level 到 start_level+num_levels-1
                test_seed_list = [start_level + (i % num_levels) for i in range(num_test_envs)]

            sample_env = make_procgen_env(
                num_envs=num_test_envs,
                env_name=env_name,
                num_levels=num_levels,
                start_level=start_level,
                distribution_mode=distribution_mode,
                seed_list=test_seed_list,
            )
            sample_res = sample_traj_multi_seed(
                int_samples, checkpoint_version, sample_env, num_envs=40, ckpt_path=checkpoint_path, eval_results=eval_results, device=device,
                filter_data=filter_data, min_correlation=min_correlation, min_increasing_ratio=min_increasing_ratio
            )
            sample_env.close()

        if not collect_train:
            if relabel_reward:
                save_path = f"{args['data_dir']}/{env_name}/test_traj-relabel-new.npz"
            else:
                save_path = f"{args['data_dir']}/{env_name}/test_traj-official.npz"
        else:
            if relabel_reward:
                save_path = f"{args['data_dir']}/{env_name}/train_traj-relabel-new.npz"
            else:
                save_path = f"{args['data_dir']}/{env_name}/train_traj-official.npz"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 使用不压缩格式保存以加快速度（数据已经在内存中压缩过了）
        logger.info(f"Saving data to {save_path}...")
        import time
        start_time = time.time()
        if 'values' in sample_res:
            np.savez(
                save_path,
                observations=sample_res['observations'],
                actions=sample_res['actions'],
                rewards=sample_res['rewards'],
                dones=sample_res['dones'],
                values=sample_res['values']
            )
        else:
            np.savez(
                save_path,
                observations=sample_res['observations'],
                actions=sample_res['actions'],
                rewards=sample_res['rewards'],
                dones=sample_res['dones']
            )
        elapsed = time.time() - start_time
        file_size = os.path.getsize(save_path)
        logger.info(f"Saved trajectory data to {save_path}, size: {file_size / (1024**2):.2f} MB, took {elapsed:.2f} seconds")
        env.close()
    else:
        logger.warning("Fixed seed mode not yet implemented for data collection")
