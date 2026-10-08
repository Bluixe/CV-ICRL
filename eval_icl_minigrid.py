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
from nets.net import MinigridTransformer, MinigridMultiheadTransformer

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContextBuffer:
    """用于存储最近的交互历史的缓冲区"""
    
    def __init__(self, horizon, state_dim, action_dim):
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 初始化缓冲区
        self.states = deque(maxlen=horizon)
        self.actions = deque(maxlen=horizon-1)
        self.rewards = deque(maxlen=horizon-1)
        
    def add(self, state, action, reward, first_step=False):
        """添加一个新的交互到缓冲区"""
        if first_step:
            self.states.append(state)
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
        
    def get_context(self):
        """获取当前缓冲区中的所有交互作为上下文"""
        # 如果缓冲区未满，用零填充
        padding_size = self.horizon - len(self.actions)
        
        if len(self.states) == 1:
            # 如果缓冲区为空，创建全零数组
            context_states = np.array([self.states[0]])  # [1, 3, 7, 7]
            context_actions = np.zeros((1,1))
            context_rewards = np.zeros((1,1))
        elif padding_size > 0:
            # 创建填充
            # padding_states = np.zeros((padding_size, 3, 7, 7))
            padding_actions = np.zeros(1)
            padding_rewards = np.zeros(1)
            
            # 将deque中的元素转换为numpy数组，确保保持正确的形状
            # states_array = np.stack(list(self.states))
            
            # 合并填充和实际数据
            context_states = np.stack(list(self.states))
            context_actions = np.concatenate([np.array(list(self.actions)), padding_actions])
            context_rewards = np.concatenate([np.array(list(self.rewards)), padding_rewards])
        else:
            # 如果缓冲区已满，直接使用所有数据
            padding_actions = np.zeros(1)
            padding_rewards = np.zeros(1)
            
            # 将deque中的元素转换为numpy数组，确保保持正确的形状
            # states_array = np.stack(list(self.states))
            
            # 合并填充和实际数据
            context_states = np.stack(list(self.states))
            context_actions = np.concatenate([np.array(list(self.actions)), padding_actions])
            context_rewards = np.concatenate([np.array(list(self.rewards)), padding_rewards])
            
        # 转换为张量
        context_states = torch.tensor(context_states, dtype=torch.float32).to(device)
        context_actions = torch.tensor(context_actions, dtype=torch.long).to(device)
        context_rewards = torch.tensor(context_rewards, dtype=torch.float32).to(device)

        context_states = context_states.unsqueeze(0)  # [1, seq_len, channels*height*width]
        
        if len(context_actions.shape) == 1:  # [seq_len]
            context_actions = context_actions.unsqueeze(0)  # [1, seq_len]

        if len(context_rewards.shape) == 1:  # [seq_len]
            context_rewards = context_rewards.unsqueeze(-1)  # [seq_len, 1]
            context_rewards = context_rewards.unsqueeze(0)  # [1, seq_len, 1]
        
        return {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_rewards': context_rewards,
        }

def preprocess_obs(obs):
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        obs = np.transpose(obs, (2, 0, 1))
    
        
    return obs

def make_env(env_name, seed=-1):  
    def _init():  
        env = gym.make(env_name, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        if seed != -1:
            env.reset(seed=seed)
        else:
            env.reset()
        return Monitor(env)
    return _init  

def make_minigrid_env(env_name, num_envs, seed_list=None):
    if seed_list is None:
        seed_list = [-1] * num_envs
    assert num_envs == len(seed_list), "Number of environments must match number of seeds"
    
    envs = [make_env(env_name, seed=seed_list[i]) for i in range(num_envs)]
    env = DummyVecEnv(envs)
    return env


def eval_models_multi_seed(model, env, horizon, num_steps=2000, context_value=False, wo_reward=False):
    """并行评估ICL模型在多个环境中的表现"""
    
    num_envs = env.num_envs

    model_name = model.__class__.__name__
    
    # 为每个环境创建上下文缓冲区
    state_dim = 3 * 7 * 7  # 假设观察是7x7的RGB图像
    action_dim = env.action_space.n
    context_buffers = [ContextBuffer(horizon, state_dim, action_dim) for _ in range(num_envs)]
    
    total_rewards = np.zeros(num_envs)
    episode_counts = np.zeros(num_envs)
    
    # 重置环境
    obs = env.reset()
    # 预处理观察
    # 确保观察的形状是 (channels, height, width)
    obs = np.array([preprocess_obs(o) for o in obs])
    for i in range(num_envs):
        # 初始化上下文缓冲区
        context_buffers[i].add(obs[i], 0, 0, first_step=True)
    
    # 打印观察的形状，用于调试
    logger.info(f"观察形状: {obs.shape}")
    
    dones = np.array([False] * num_envs)
    context_values = np.zeros(num_envs, dtype=np.float32)
    for t in tqdm(range(num_steps), desc="评估进度", ncols=100):
        # 为每个环境准备上下文和动作
        actions = np.zeros(num_envs, dtype=np.int64)  # 默认动作为0
        labels = np.zeros(num_envs, dtype=np.float32)
        # 
        
        for i in range(num_envs):
            # 获取上下文
            context = context_buffers[i].get_context()
            context_len = context['context_states'].shape[1]
            # print(context_len)
            
            # 使用模型预测动作
            with torch.no_grad():
                if model_name == 'MinigridTransformer' or model_name == 'MinigridDPTTransformer' or model_name == 'MinigridTargetTransformer':
                    action_probs = model(context)
                elif model_name == 'MinigridMultiheadTransformer':
                    action_probs, label = model(context)
                    # logger.info(label)
                elif model_name == 'MinigridSICQLTransformer':
                    action_probs, _, _ = model(context)
                else:
                    raise ValueError(f"Unsupported model type: {model_name}")
                # 如果模型输出是logits，转换为概率
                # print(action_probs)
                if len(action_probs.shape) == 3:
                    action_probs = action_probs[0, context_len-1]  # 取最后一个时间步的预测
                    if model_name == 'MinigridMultiheadTransformer':
                        label = label[0, context_len-1]
                action = torch.multinomial(torch.softmax(action_probs, dim=-1), num_samples=1).item()
                actions[i] = action
                if model_name == 'MinigridMultiheadTransformer':
                    labels[i] = label.item()

        # 执行动作
        next_obs, rewards, dones, infos = env.step(actions)

        if context_value:
            if model_name == 'MinigridMultiheadTransformer':
                context_values = np.maximum(context_values, labels)
                print(context_values)
            else:
                # context_values = np.ones_like(rewards) * (min(t, 700) / 800)
                context_values = np.ones_like(rewards)
        
        # 预处理下一个观察
        # 确保观察的形状是 (channels, height, width)
        next_obs = np.array([preprocess_obs(o) for o in next_obs])
        
        # 更新上下文缓冲区和统计信息
        for i in range(num_envs):
            if context_value:
                context_buffers[i].add(next_obs[i], actions[i], context_values[i])
            else:
                context_buffers[i].add(next_obs[i], actions[i], rewards[i])
            total_rewards[i] += rewards[i]
            if dones[i]:  # 如果环境刚刚完成
                episode_counts[i] += 1
                # # 重置上下文缓冲区
                # context_buffers[i] = ContextBuffer(horizon, state_dim, action_dim)
        
        # 更新观察
        obs = next_obs

        wandb.log({"culmulative reward": sum(total_rewards),
                   "episode avg reward": sum(total_rewards)/sum(episode_counts)})
    
    # 计算平均奖励
    valid_episodes = episode_counts > 0
    if np.any(valid_episodes):
        avg_rewards = total_rewards[valid_episodes] / episode_counts[valid_episodes]
        avg_reward = np.mean(avg_rewards)
    else:
        avg_reward = 0.0
    
    logger.info(f"Total episodes completed: {np.sum(episode_counts)}")
    logger.info(f"Average reward per episode: {avg_reward}")
    
    return avg_reward

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
    parser.add_argument('--multi_env', action='store_true', default=False,
                        help='Use multiple environments or not')
    parser.add_argument('--eval_env', type=str, default='MiniGrid-LavaCrossingS9N3-v0',
                        help='Evaluation environment name')
    parser.add_argument('--context_value', action='store_true', default=False,
                        help='Relabel reward for training or not')
    parser.add_argument('--predict_value', action='store_true', default=False,
                        help='Use automatic relabel for training or not')
    
    args = vars(parser.parse_args())
    
    # 设置随机种子
    seed = args['seed']
    seed_list = [i for i in range(args['num_envs'])]
    env_name = args['env']
    algorithm = args['algorithm']
    context_value = args['context_value']
    predict_value = args['predict_value']
    
    # 创建环境
    eval_env_name = args["eval_env"]
    env = make_minigrid_env(eval_env_name, args['num_envs'], seed_list)
    
    # 设置模型配置
    config = {
        'horizon': args['H'],
        'state_dim': 2,  # MiniGrid的状态维度
        'action_dim': env.action_space.n,
        'n_layer': args['layer'],
        'n_embd': args['embd'],
        'n_head': args['head'],
        'shuffle': args['shuffle'],
        'dropout': args['dropout'],
        'test': True,
        'store_gpu': True,
        'image_size': 7,  # MiniGrid的图像大小
    }
    logger.info(config)

    env_name_1 = env_name.replace("MiniGrid-", "").replace("-v0", "")
    eval_env_name_1 = eval_env_name.replace("MiniGrid-", "").replace("-v0", "")

    if not args['multi_env']:
        if args['context_value']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-context_value"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-context_value"
        elif args['predict_value']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-predict_value"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-predict_value"
        else:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}"
    elif args['multi_env']:
        if args['context_value']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_context_value"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_context_value"
        elif args['predict_value']:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_predict_value"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env_predict_value"
        else:
            wandb_name=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env"
            wandb_group=f"eval-{env_name_1}-{eval_env_name_1}-{algorithm}-multi_env"
    else:
        raise ValueError("Unsupported configuration for wandb name and group")

    run = wandb.init(
        project="eval-minigrid-icl",
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        )
    
    # 加载模型
    if algorithm == 'ad':
        if args['multi_env']:
            if args['context_value']:
                model_path = f'models/MiniGrid/{args["env"]}/{algorithm}/epoch15_multi_env_context_value.pt'
                model = MinigridTransformer(config, mode="ad").to(device)
            elif args['predict_value']:
                model_path = f'models/MiniGrid/{args["env"]}/{algorithm}/epoch15_multi_env_predict_value.pt'
                model = MinigridMultiheadTransformer(config, mode="ad").to(device)
            else:
                model_path = f'models/MiniGrid/{args["env"]}/{algorithm}/epoch15_multi_env.pt'
                model = MinigridTransformer(config, mode="ad").to(device)
        else:
            if args['context_value']:
                model_path = f'models/MiniGrid/{args["env"]}/{algorithm}/epoch15_context_value.pt'
                model = MinigridTransformer(config, mode="ad").to(device)
            elif args['predict_value']:
                model_path = f'models/MiniGrid/{args["env"]}/{algorithm}/epoch15_predict_value.pt'
                model = MinigridMultiheadTransformer(config, mode="ad").to(device)
            else:
                model_path = f'models/MiniGrid/{args["env"]}/{algorithm}/epoch15_more.pt'
                model = MinigridTransformer(config, mode="ad").to(device)
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
        context_value=(context_value or predict_value),
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