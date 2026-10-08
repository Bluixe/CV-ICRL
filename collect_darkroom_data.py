from envs import darkroom_env, bandit_env
from envs.darkroom_env import DarkroomEnvVec
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback
import minigrid
import time

import argparse
import os
import pickle
import random

import gymnasium as gym # gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from skimage.transform import resize
from IPython import embed
import common_args
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import ImgObsWrapper

from loguru import logger
from env_list import ENVS, HARD_ENVS

from multiprocessing import Process, Queue, set_start_method
import multiprocessing

import torch

# 设置多进程启动方法为 'spawn'，解决 CUDA 在 fork 子进程中的问题
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # 如果已经设置过，则忽略错误


GOALS = [
    (2, 7),
    (8, 3),
    (1, 4),
    (3, 6),
    (7, 8),
    (4, 2),
    (6, 5),
    (1, 7),
    (8, 4),
    (2, 1),
    (5, 3),
    (3, 8),
    (7, 2),
    (6, 1),
    (8, 7),
    (1, 3),
    (2, 6),
    (5, 8),
    (7, 1),
    (3, 2),
]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_traj_multi_seed(model_idxs, ckpt_path, env, num_envs, slice_idx):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs = env.reset()
    episode_obs = []
    episode_actions = []
    episode_rewards = []
    episode_lambda = []
    episode_dones = []
    model_num = len(model_idxs)
    length_per_model = 100
    model_per_traj = 4
    total_num = model_num

    # eval_final_model
    model_path = f"{ckpt_path}/ppo_1000000_steps.zip"
    model = PPO.load(model_path, env=env)
    eval_rewards = []
    eval_rewards_by_model = [-1 for _ in range(model_num)]
    for i in range(length_per_model):
        action, _ = model.predict(obs, deterministic=False)
        new_obs, reward, done, info = env.step(action)
        eval_rewards.append(reward)
        obs = new_obs

    final_eval_reward = np.sum(eval_rewards) / num_envs 

    for offset in range(model_num):

        for iter_idx in range(offset, offset + model_per_traj):

            idx = min(iter_idx, model_num - 1)
            model_path = f"{ckpt_path}/ppo_{model_idxs[idx]}_steps.zip"

            model = PPO.load(model_path, env=env)
            logger.info(f"Load model from {model_path}")
            logger.info(f"Idx {idx}, offset {offset+1}/{model_num - model_per_traj}")
            
            eval_rewards = []
            for i in range(length_per_model):
                action, _ = model.predict(obs, deterministic=False)
                new_obs, reward, done, info = env.step(action) # return ndarray
                eval_rewards.append(reward)
                obs = new_obs

            if eval_rewards_by_model[idx] == -1:
                eval_rewards_by_model[idx] = np.sum(eval_rewards) / num_envs
                print("changing eval_rewards_by_model")
                if idx > 0:
                    eval_rewards_by_model[idx] = max(eval_rewards_by_model[idx], eval_rewards_by_model[idx-1])

            eval_reward = np.sum(eval_rewards) / num_envs
            print(f"Eval reward: {eval_reward}, max reward: {eval_rewards_by_model[idx]}, Final eval reward: {final_eval_reward}")
            
            for i in range(length_per_model):
                action, _ = model.predict(obs, deterministic=False)
                new_obs, reward, done, info = env.step(action) # return ndarray

                # relabeled_reward = np.ones_like(reward) * eval_rewards_by_model[idx]
                relabeled_reward = np.ones_like(reward) * (min(length_per_model * iter_idx + i, length_per_model * model_num)) / (length_per_model * model_num)
                episode_obs.append(obs)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_lambda.append(relabeled_reward)
                episode_dones.append(done)
                obs = new_obs
    
    trajectory = {
        "observations": np.stack(episode_obs),
        "actions": np.array(episode_actions),
        "rewards": np.array(episode_rewards),
        "lambda": np.array(episode_lambda),
        "dones": np.array(episode_dones),
    }
    # assert len(model_idxs) % 8 == 0, "model_idxs should be divisible by 8"
    trajectory['observations'] = trajectory['observations'].transpose((1, 0, 2)).reshape((num_envs*total_num, length_per_model*model_per_traj, 2))
    trajectory['actions'] = trajectory['actions'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
    trajectory['rewards'] = trajectory['rewards'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
    trajectory['lambda'] = trajectory['lambda'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
    trajectory['dones'] = trajectory['dones'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
    logger.info(f"obs: {trajectory['observations'].shape}")
    logger.info(f"act: {trajectory['actions'].shape}")

    return trajectory

def make_env(dim, goal, horizon):  
    def _init():  
        env = darkroom_env.DarkroomEnv(dim, goal, horizon)
        return Monitor(env)
        # return env
    return _init  

def make_darkroom_env(goal, dim, horizon, num_envs):
    envs = [make_env(dim, goal, horizon) for _ in range(num_envs)]
    env = DummyVecEnv(envs)
    return env


def process_slice(goal, model_idxs, temp_dir, ckpt_path, num_per_slice, slice_idx):
    """处理单个 slice 的数据收集任务，并将结果保存到临时文件"""
    try:
        sample_env = make_darkroom_env(goal, 9, 100, num_per_slice)
        sample_res = sample_traj_multi_seed(
            model_idxs, ckpt_path, sample_env, num_envs=num_per_slice, slice_idx=slice_idx)
        
        # 将结果保存到临时文件
        temp_file = os.path.join(temp_dir, f"slice_{slice_idx}.pkl")
        with open(temp_file, "wb") as f:
            pickle.dump(sample_res, f)
        return True
    except Exception as e:
        logger.error(f"Error processing slice {slice_idx}: {str(e)}")
        return False

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    parser.add_argument("--fix_seed", action="store_true",
                        default=False, help="Fix seed or not")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--cnn", action="store_true",
                        default=False, help="Use CnnPolicy or not")
    parser.add_argument("--train", action="store_true",
                        default=False, help="Collect train data or not")
    parser.add_argument("--context_value", action="store_true",
                        default=False, help="Relabel reward or not")
    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_eval_envs = args['envs_eval']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    var = args['var']
    cov = args['cov']
    env_id_start = args['env_id_start']
    env_id_end = args['env_id_end']
    lin_d = args['lin_d']
    context_value = args['context_value']

    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }

    config.update({'dim': dim, 'rollin_type': 'uniform'})

    assert args['cnn'], "Only support cnn models"
        
    # Step1: eval_checkpoints_and_get_avg_return
    checkpoint_paths = []
    for goal in GOALS:
        goal_x, goal_y = goal
        checkpoint_path = f"models/Darkroom/{goal_x}-{goal_y}"
        checkpoint_dir = os.listdir(checkpoint_path)[0]
        checkpoint_path = f"{checkpoint_path}/{checkpoint_dir}"
        checkpoint_paths.append(checkpoint_path)
    
    idxs = [50000*i for i in range(1, 21)]

    collect_train = args["train"]
    if collect_train:
        sample_env_num = 2000
        slices = 20
    else:
        sample_env_num = 100
        slices = 10
    num_per_slice = sample_env_num // slices
    
    # 创建临时目录保存结果
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="minigrid_data_")
    logger.info(f"Created temporary directory for results: {temp_dir}")
    
    # 创建并启动进程
    processes = []
    for slice_idx in range(slices):
        p = Process(
            target=process_slice,
            args=(GOALS[slice_idx], idxs, temp_dir, checkpoint_paths[slice_idx], num_per_slice, slice_idx)
        )
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 从临时文件中加载结果
    sample_res_list = []
    for slice_idx in range(slices):
        temp_file = os.path.join(temp_dir, f"slice_{slice_idx}.pkl")
        if os.path.exists(temp_file):
            try:
                with open(temp_file, "rb") as f:
                    sample_res = pickle.load(f)
                    sample_res_list.append(sample_res)
                    logger.info(f"Loaded results from {temp_file}")
            except Exception as e:
                logger.error(f"Error loading results from {temp_file}: {str(e)}")
        else:
            logger.warning(f"Results for slice {slice_idx+1}/{slices} not found")
    
    # 检查是否有足够的结果进行合并
    if len(sample_res_list) == 0:
        logger.error("No results collected from any process!")
        raise RuntimeError("Failed to collect any results from processes")
    
    # 合并所有 slice 的结果
    logger.info(f"Merging results from {len(sample_res_list)} slices")
    try:
        if context_value:
            sample_res = {
                "observations": np.concatenate([res['observations'] for res in sample_res_list], axis=0),
                "actions": np.concatenate([res['actions'] for res in sample_res_list], axis=0),
                "lambda": np.concatenate([res['lambda'] for res in sample_res_list], axis=0),
                "rewards": np.concatenate([res['rewards'] for res in sample_res_list], axis=0),
                "dones": np.concatenate([res['dones'] for res in sample_res_list], axis=0),
            }
        else:
            sample_res = {
                "observations": np.concatenate([res['observations'] for res in sample_res_list], axis=0),
                "actions": np.concatenate([res['actions'] for res in sample_res_list], axis=0),
                "rewards": np.concatenate([res['rewards'] for res in sample_res_list], axis=0),
                "dones": np.concatenate([res['dones'] for res in sample_res_list], axis=0),
            }
        logger.info(f"obs: {sample_res['observations'].shape}")
        logger.info(f"act: {sample_res['actions'].shape}")
    except Exception as e:
        logger.error(f"Error merging results: {str(e)}")
        raise
    
    if context_value:
        if not collect_train:
            save_path = f"datasets/Darkroom/test_traj_context_value_more.pkl"
        else:
            save_path = f"datasets/Darkroom/train_traj_context_value_more.pkl"
    else:
        if not collect_train:
            save_path = f"datasets/Darkroom/test_traj_more.pkl"
        else:
            save_path = f"datasets/Darkroom/train_traj_more.pkl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(sample_res, f)

            



        

    
