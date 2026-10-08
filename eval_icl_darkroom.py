import os
from envs import darkroom_env, bandit_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import wandb
os.environ.setdefault("WANDB_MODE", "disabled")
from wandb.integration.sb3 import WandbCallback
import minigrid
from tqdm import tqdm

import argparse
import os
import pickle
import random
from collections import deque

import gymnasium as gym # gym
import numpy as np
import torch
from skimage.transform import resize
from IPython import embed
import common_args
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import ImgObsWrapper
from loguru import logger
import pprint
from env_list import ENVS, HARD_ENVS
from nets.net import DarkroomTransformer, DarkroomLambdaTransformer
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GOALS = [
    (4, 3), (5, 4), (4, 6), (5, 1), (5, 7), (8, 6), (2, 2), (1, 6), (7, 4), (7, 7), 
    (6, 8), (4, 5), (4, 8), (8, 8), (6, 4), (4, 7), (3, 5), (4, 4), (2, 5), (7, 5)
]

class ContextBuffer:
    """用于存储最近的交互历史的缓冲区"""
    
    def __init__(self, horizon, action_dim, context_value=False):
        self.horizon = horizon
        self.action_dim = action_dim
        self.context_value = context_value
        
        # 初始化缓冲区
        self.states = deque(maxlen=horizon)
        self.actions = deque(maxlen=horizon-1)
        self.rewards = deque(maxlen=horizon-1)
        if context_value:
            self.lambda_ = deque(maxlen=horizon-1)
        
        
    def add(self, state, action, reward, lambda_=None, first_step=False):
        """添加一个新的交互到缓冲区"""
        if first_step:
            self.states.append(state)
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            if self.context_value:
                self.lambda_.append(lambda_)
        
    def get_context(self):
        """获取当前缓冲区中的所有交互作为上下文"""
        # 如果缓冲区未满，用零填充
        padding_size = self.horizon - len(self.actions)
        
        if len(self.states) == 1:
            # 如果缓冲区为空，创建全零数组
            context_states = np.array([self.states[0]])  # [1, 3, 7, 7]
            context_actions = np.zeros((1,1))
            context_rewards = np.zeros((1,1))
            context_lambdas = np.zeros((1,1)) if self.context_value else None
        elif padding_size > 0:
            # 创建填充
            # padding_states = np.zeros((padding_size, 3, 7, 7))
            padding_actions = np.zeros(1)
            padding_rewards = np.zeros(1)
            padding_lambdas = np.zeros(1) if self.context_value else None
            
            # 合并填充和实际数据
            context_states = np.stack(list(self.states))
            context_actions = np.concatenate([np.array(list(self.actions)), padding_actions])
            context_rewards = np.concatenate([np.array(list(self.rewards)), padding_rewards])
            context_lambdas = np.concatenate([np.array(list(self.lambda_)), padding_lambdas]) if self.context_value else None
        else:
            # 如果缓冲区已满，直接使用所有数据
            padding_actions = np.zeros(1)
            padding_rewards = np.zeros(1)
            padding_lambdas = np.zeros(1) if self.context_value else None
            
            # 合并填充和实际数据
            context_states = np.stack(list(self.states))
            context_actions = np.concatenate([np.array(list(self.actions)), padding_actions])
            context_rewards = np.concatenate([np.array(list(self.rewards)), padding_rewards])
            context_lambdas = np.concatenate([p.array(list(self.lambda_))], padding_lambdas) if self.context_value else None
            
        # 转换为张量
        context_states = torch.tensor(context_states, dtype=torch.float32).to(device)
        context_actions = torch.tensor(context_actions, dtype=torch.long).to(device)
        context_rewards = torch.tensor(context_rewards, dtype=torch.float32).to(device)

        if self.context_value:
            context_lambdas = torch.tensor(context_lambdas, dtype=torch.float32).to(device)

        context_states = context_states.unsqueeze(0)  # [1, seq_len, channels*height*width]
        
        # context_actions 应该是 [batch_size, seq_len]
        if len(context_actions.shape) == 1:  # [seq_len]
            context_actions = context_actions.unsqueeze(0)  # [1, seq_len]
        
        # context_rewards 应该是 [batch_size, seq_len, 1]
        if len(context_rewards.shape) == 1:  # [seq_len]
            context_rewards = context_rewards.unsqueeze(-1)  # [seq_len, 1]
            context_rewards = context_rewards.unsqueeze(0)  # [1, seq_len, 1]
        
        if self.context_value and len(context_lambdas.shape) == 1:
            context_lambdas = context_lambdas.unsqueeze(-1)
            context_lambdas = context_lambdas.unsqueeze(0)  # [1,
        
        if self.context_value:
            return {
                'context_states': context_states,
                'context_actions': context_actions,
                'context_rewards': context_rewards,
                'context_lambda': context_lambdas
            }
        else:
            return {
                'context_states': context_states,
                'context_actions': context_actions,
                'context_rewards': context_rewards
            }


def make_env(dim, goal, horizon):  
    def _init():  
        env = darkroom_env.DarkroomEnv(dim, goal, horizon)
        return Monitor(env)
        # return env
    return _init  

def make_darkroom_env(goals, dim, horizon, num_envs):
    assert len(goals) == num_envs, "Number of goals must match number of environments"
    envs = [make_env(dim, goals[i], horizon) for i in range(num_envs)]
    env = DummyVecEnv(envs)
    return env

def eval_models_multi_seed(model, env, horizon, num_steps=2000, context_value=False, wo_reward=False):
    """并行评估ICL模型在多个环境中的表现"""
    
    num_envs = env.num_envs
    model_name = model.__class__.__name__
    
    # 为每个环境创建上下文缓冲区
    action_dim = 5
    context_buffers = [ContextBuffer(horizon, action_dim, context_value) for _ in range(num_envs)]
    
    total_rewards = np.zeros(num_envs)
    episode_counts = np.zeros(num_envs)
    
    # 重置环境
    obs = env.reset()
    # 预处理观察
    # 确保观察的形状是 (channels, height, width)
    obs = np.array([o for o in obs])
    for i in range(num_envs):
        # 初始化上下文缓冲区
        context_buffers[i].add(obs[i], 0, 0, first_step=True)
    
    # 打印观察的形状，用于调试
    logger.info(f"观察形状: {obs.shape}")
    
    dones = np.array([False] * num_envs)

    max_rewards = np.array([0] * num_envs)  # 假设每个环境的最大奖励为100
    episode_rewards = np.zeros(num_envs)  # 用于跟踪每个环境的当前episode奖励
    episode_trajs = [[] for _ in range(num_envs)]  # 用于存储每个环境的轨迹
    
    for t in tqdm(range(num_steps), desc="评估进度", ncols=100):
        # 为每个环境准备上下文和动作
        actions = np.zeros(num_envs, dtype=np.int64)  # 默认动作为0
        
        for i in range(num_envs):
            # 获取上下文
            context = context_buffers[i].get_context()
            context_len = context['context_states'].shape[1]
            # print(context_len)
            
            # 使用模型预测动作
            with torch.no_grad():
                if model_name in ['DarkroomTransformer', 'DarkroomLambdaTransformer']:
                    action_probs = model(context)
                elif model_name == 'MinigridMultiheadTransformer':
                    action_probs, _ = model(context)
                else:
                    raise ValueError(f"Unsupported model type: {model_name}")
                # 如果模型输出是logits，转换为概率
                if len(action_probs.shape) == 3:
                    action_probs = action_probs[0, context_len-1]  # 取最后一个时间步的预测
                # action = torch.argmax(action_probs).item()
                # 采样动作
                action = torch.multinomial(torch.softmax(action_probs, dim=-1), num_samples=1).item()
                actions[i] = action
        
        # 执行动作
        next_obs, rewards, dones, infos = env.step(actions)

        if context_value:
            # context_values = np.ones_like(rewards) * min(t, 3200) / 3200
            context_values = np.ones_like(rewards) * (min(t, 1200) / 1200)
        
        # 预处理下一个观察
        # 确保观察的形状是 (channels, height, width)
        next_obs = np.array([o for o in next_obs])
        
        # 更新上下文缓冲区和统计信息
        for i in range(num_envs):
            if context_value:
                context_buffers[i].add(next_obs[i], actions[i], rewards[i], context_values[i])
                # context_buffers[i].add(next_obs[i], actions[i], rewards[i], max_rewards[i])
            elif wo_reward:
                context_buffers[i].add(next_obs[i], actions[i], 0)
            else:
                context_buffers[i].add(next_obs[i], actions[i], rewards[i])
            total_rewards[i] += rewards[i]
            episode_rewards[i] += rewards[i]
            episode_trajs[i].append(obs[i])
            if dones[i]:
                episode_counts[i] += 1
                final_obs = obs[i]
                max_rewards[i] = max(max_rewards[i], episode_rewards[i])
                wandb.log({
                    f"Env {i} Reward": episode_rewards[i],
                    f"Env {i} Max Reward": max_rewards[i],
                })
                episode_trajs[i] = []
                episode_rewards[i] = 0
                
        obs = next_obs

        wandb.log({"culmulative reward": sum(total_rewards)})
    
    valid_episodes = episode_counts > 0
    if np.any(valid_episodes):
        avg_rewards = total_rewards[valid_episodes] / episode_counts[valid_episodes]
        avg_reward = np.mean(avg_rewards)
    else:
        avg_reward = 0.0
    
    logger.info(f"Total episodes completed: {np.sum(episode_counts)}")
    logger.info(f"Average reward per episode: {avg_reward}")
    
    return 

def main():
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    
    parser.add_argument('--algorithm', type=str, default='ad', choices=['ad', 'dpt'], help='算法类型')
    parser.add_argument('--num_envs', type=int, default=8, help='并行环境数量')
    parser.add_argument('--num_steps', type=int, default=1000, help='评估的episode数量')
    parser.add_argument('--fix_seed', action='store_true', help='是否固定随机种子')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--use_value', action='store_true', default=False,
                    help='Use value function or not')
    parser.add_argument('--use_finetune', action='store_true', default=False,
                        help='Use finetune data or not')
    parser.add_argument('--multi_env', action='store_true', default=False,
                        help='Use multiple environments or not')
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
    parser.add_argument('--context_value', action='store_true', default=False,
                        help='Relabel reward for training or not')
    
    args = vars(parser.parse_args())
    
    # 设置随机种子
    seed = args['seed']
    seed_list = [i for i in range(args['num_envs'])]
    env_name = args['env']
    algorithm = args['algorithm']
    use_value = args['use_value']
    pred_reward = args['pred_reward']
    reward_mode = args['reward_mode']
    context_value = args['context_value']
    
    # 设置模型配置
    config = {
        'horizon': args['H'],
        'state_dim': 2,
        'action_dim': 5,
        'n_layer': args['layer'],
        'n_embd': args['embd'],
        'n_head': args['head'],
        'shuffle': args['shuffle'],
        'dropout': args['dropout'],
        'test': True,
        'store_gpu': True,
        'image_size': 7,
    }
    logger.info(config)


    env = make_darkroom_env(GOALS, 9, 100, len(GOALS))

    if args['context_value']:
        wandb_name=f"eval-Darkroom-{algorithm}-context_value"
        wandb_group=f"eval-Darkroom-{algorithm}-context_value"
    else:
        wandb_name=f"eval-Darkroom-{algorithm}"
        wandb_group=f"eval-Darkroom-{algorithm}"

    run = wandb.init(
        project="eval-darkroom-icl",
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        )
    
    # 加载模型
    if algorithm == 'ad':
        if args['context_value']:
            model_path = f'models/Darkroom/{algorithm}/epoch15_context_value_more.pt'
            model = DarkroomLambdaTransformer(config, mode="ad").to(device)
        else:
            model_path = f'models/Darkroom/{algorithm}/epoch15_more.pt'
            model = DarkroomTransformer(config, mode="ad").to(device)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
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
        context_value=context_value
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
    
    return results

if __name__ == '__main__':
    main()