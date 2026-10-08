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

# ENV = "MiniGrid-LavaCrossingS9N2-v0"
ENV = "MiniGrid-LavaCrossingS9N3-v0"
# ENV = "MiniGrid-SimpleCrossingS9N3-v0"
# ENV = "MiniGrid-SimpleCrossingS11N5-v0"
# ENV = "MiniGrid-RedBlueDoors-8x8-v0"
# ENV = "MiniGrid-BlockedUnlockPickup-v0"
# ENV = "MiniGrid-UnlockPickup-v0"
# ENV = "MiniGrid-Unlock-v0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_models(model, env, seed_list):

    total_reward = 0
    for seed in seed_list:
        obs, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=False)
            new_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs = new_obs
    
    return total_reward / len(seed_list)

def eval_models_multi_seed(model, env):

    total_reward = 0
    obs = env.reset()
    eps = 0
    for i in range(500):
        action, _ = model.predict(obs, deterministic=False)
        new_obs, reward, done, info = env.step(action)
        # print(done)
        total_reward += np.sum(reward)
        obs = new_obs
        eps += np.sum(done)

    logger.info(f"num_eps: {eps}")
    if eps == 0:
        return 0
    else:
        return total_reward / eps

def sample_traj_multi_seed(model_idxs, ckpt_version, env, num_envs, ckpt_path, use_q_net=False, eval_results=None):
    
    # trajectories = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs = env.reset()
    episode_obs = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    episode_values = []
    model_num = len(model_idxs)
    length_per_model = 50
    model_per_traj = 8
    max_offset = model_per_traj // 2
    total_num = (len(model_idxs) - model_per_traj) // 2 + 1
    if use_q_net:
        from nets.cql.discrete_cql import DiscreteCQLPolicy
        from nets.cql.nets import QValueNet
        config = {
            "dropout": 0,
            'n_embd': 256
        }
        q_net = QValueNet(
            obs_space=7,
            action_space=7,
            num_quantiles=200,
            device=device,
            args=config,
        )
        optim = torch.optim.Adam(q_net.parameters(), lr=1e-4)
        q_policy = DiscreteCQLPolicy(
            model=q_net,
            optim=optim,
            action_space=Discrete(7),
            discount_factor=0.99,
            num_quantiles=200,
            estimation_step=1,
            target_update_freq=500,
            min_q_weight=10.0
        ).to(device)

    for offset in range(max_offset):

        max_idx = model_num - (model_num % model_per_traj)
        if max_idx + offset * 2 > model_num:
            max_idx = max_idx - model_per_traj
        else:
            max_idx = max_idx
        for idx in range(max_idx):
            model_path = f"{ckpt_path}/ppo_{ckpt_version[model_idxs[idx+offset*2]]}_steps.zip"
            if eval_results is not None:
                result = eval_results[model_idxs[idx+offset*2]]
            model = PPO.load(model_path, env=env)
            logger.info(f"Load model from {model_path}")
            logger.info(f"Idx {idx+1} of total {max_idx}, offset {offset+1}/{max_offset}, totally {total_num} models")
            
            for i in range(length_per_model):
                action, _ = model.predict(obs, deterministic=False)
                # logger.info(obs.shape)
                if use_q_net:
                    q_value = q_policy.get_q_value(obs.transpose(0, 3, 1, 2), action, device)
                    q_value = q_value.cpu().detach().numpy()
                    # logger.info(f"q_value shape: {q_value.shape}")
                    episode_values.append(q_value)

                # begin_time = time.time()
                new_obs, reward, done, info = env.step(action) # return ndarray
                # 记录轨迹数据
                episode_obs.append(obs)
                episode_actions.append(action)
                if eval_results is None:
                    episode_rewards.append(reward)
                else:
                    episode_rewards.append(np.ones(reward.shape) * result)
                
                episode_dones.append(done)

                obs = new_obs
    
    # total_num * length_per_model * model_per_traj, num_envs, ... 
    if not use_q_net:
        trajectory = {
            "observations": np.stack(episode_obs),
            "actions": np.array(episode_actions),
            "rewards": np.array(episode_rewards),
            "dones": np.array(episode_dones),
        }
        # assert len(model_idxs) % 8 == 0, "model_idxs should be divisible by 8"
        trajectory['observations'] = trajectory['observations'].transpose((1, 0, 4, 2, 3)).reshape((num_envs*total_num, length_per_model*model_per_traj, 3, 7, 7))
        trajectory['actions'] = trajectory['actions'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
        trajectory['rewards'] = trajectory['rewards'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
        trajectory['dones'] = trajectory['dones'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
    else:
        trajectory = {
            "observations": np.stack(episode_obs),
            "actions": np.array(episode_actions),
            "rewards": np.array(episode_rewards),
            "dones": np.array(episode_dones),
            "values": np.array(episode_values),
        }
        # assert len(model_idxs) % 8 == 0, "model_idxs should be divisible by 8"
        trajectory['observations'] = trajectory['observations'].transpose((1, 0, 4, 2, 3)).reshape((num_envs*total_num, length_per_model*model_per_traj, 3, 7, 7))
        trajectory['actions'] = trajectory['actions'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
        trajectory['rewards'] = trajectory['rewards'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
        trajectory['values'] = trajectory['values'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))
        trajectory['dones'] = trajectory['dones'].transpose((1, 0)).reshape((num_envs*total_num, length_per_model*model_per_traj))

    logger.info(f"obs: {trajectory['observations'].shape}")
    logger.info(f"act: {trajectory['actions'].shape}")

    return trajectory


def sample_dpt_traj_multi_seed(model_idxs, ckpt_version, env, num_envs, ckpt_path):
    
    # trajectories = []
    obs = env.reset()
    episode_obs = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    optimal_actions = []
    model_num = len(model_idxs)
    length_per_model = 50
    model_per_traj = 8
    optimal_model_path = f"{ckpt_path}/ppo_{ckpt_version[-1]}_steps.zip" 
    optimal_model = PPO.load(optimal_model_path, env=env)
    max_idx = model_num
    for idx in range(max_idx):
        model_path = f"{ckpt_path}/ppo_{ckpt_version[model_idxs[idx]]}_steps.zip"
        model = PPO.load(model_path, env=env)
        logger.info(f"Load model from {model_path}")
        logger.info(f"Idx {idx+1} of total {max_idx}")
        
        for i in range(length_per_model):
            action, _ = model.predict(obs, deterministic=False)
            optimal_action, _ = optimal_model.predict(obs, deterministic=True)

            # begin_time = time.time()
            new_obs, reward, done, info = env.step(action) # return ndarray
            # end_time = time.time()
            # print("env time: ", end_time - begin_time)
            # 记录轨迹数据
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            optimal_actions.append(optimal_action)

            obs = new_obs
    
    # total_num * length_per_model * model_per_traj, num_envs, ... 
    trajectory = {
        "observations": np.stack(episode_obs),
        "actions": np.array(episode_actions),
        "rewards": np.array(episode_rewards),
        "dones": np.array(episode_dones),
        "optimal_actions": np.array(optimal_actions),
    }
    # assert len(model_idxs) % 8 == 0, "model_idxs should be divisible by 8"
    trajectory['observations'] = trajectory['observations'].transpose((1, 0, 4, 2, 3))
    trajectory['actions'] = trajectory['actions'].transpose((1, 0))
    trajectory['rewards'] = trajectory['rewards'].transpose((1, 0))
    trajectory['dones'] = trajectory['dones'].transpose((1, 0))
    trajectory['optimal_actions'] = trajectory['optimal_actions'].transpose((1, 0))
    
    logger.info(f"obs: {trajectory['observations'].shape}")
    logger.info(f"act: {trajectory['actions'].shape}")

    return trajectory


def sample_trajectories(model, env, num_trajs=5, max_steps=1000):
    trajectories = []
    for _ in range(num_trajs):
        obs, _ = env.reset()
        terminated, truncated = False, False
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_terminateds = []
        episode_truncateds = []

        step = 0
        while not (terminated or truncated) and step < max_steps:
            action, _ = model.predict(obs, deterministic=False)
            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录轨迹数据
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_terminateds.append(terminated)
            episode_truncateds.append(truncated)
            
            obs = new_obs
            step += 1

        # 将列表转换为numpy数组
        trajectory = {
            "observations": np.stack(episode_obs),
            "actions": np.array(episode_actions),
            "rewards": np.array(episode_rewards),
            "terminateds": np.array(episode_terminateds),
            "truncateds": np.array(episode_truncateds)
        }
        trajectories.append(trajectory)
    
    return trajectories

def make_eval_env(seed=-1):
    env = gym.make(ENV, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    if seed != -1:
        logger.info(f"set seed to {seed}")
        env.reset(seed=seed)
    else:
        env.reset()

    return env

def make_env(seed=-1):  
    def _init():  
        # env = darkroom_env.DarkroomEnv(dim, goal, horizon)
        env = gym.make(ENV, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        if seed != -1:
            logger.info(f"set seed to {seed}")
            env.reset(seed=seed)
        else:
            env.reset()
        return Monitor(env)
        # return env
    return _init  

def make_minigrid_env(num_envs, seed_list):
    assert num_envs == len(seed_list)
    envs = [make_env(seed=seed_list[i]) for i in range(num_envs)]
    # env = SubprocVecEnv(envs)
    env = DummyVecEnv(envs)
    return env

def process_slice(slice_idx, slices, int_samples, checkpoint_version, temp_dir, ckpt_path, num_per_slice, eval_results=None, mode="ad", use_q_net=False):
    """处理单个 slice 的数据收集任务，并将结果保存到临时文件"""
    assert mode in ["ad", "dpt"], "mode should be either 'ad' or 'dpt'"
    try:
        logger.info(f"Slice {slice_idx+1}/{slices} started")
        sample_seed_list = [10000+i for i in range(slice_idx*num_per_slice, (slice_idx+1)*num_per_slice)]
        sample_env = make_minigrid_env(num_per_slice, sample_seed_list)
        if mode == "dpt":
            sample_res = sample_dpt_traj_multi_seed(
                int_samples, checkpoint_version, sample_env, num_envs=num_per_slice, ckpt_path=ckpt_path)
        else:
            sample_res = sample_traj_multi_seed(
                int_samples, checkpoint_version, sample_env, num_envs=num_per_slice, ckpt_path=ckpt_path, use_q_net=use_q_net, eval_results=eval_results)
        
        # 将结果保存到临时文件
        temp_file = os.path.join(temp_dir, f"slice_{slice_idx}.pkl")
        with open(temp_file, "wb") as f:
            pickle.dump(sample_res, f)
        
        logger.info(f"Slice {slice_idx+1}/{slices} completed and saved to {temp_file}")
        return True
    except Exception as e:
        logger.error(f"Error in slice {slice_idx+1}/{slices}: {str(e)}")
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
    parser.add_argument("--dpt", action="store_true",
                        default=False, help="Use DPT or not")
    parser.add_argument("--train", action="store_true",
                        default=False, help="Collect train data or not")
    parser.add_argument("--use_q_net", action="store_true",
                        default=False, help="Use QValueNet or not")
    parser.add_argument("--context_value", action="store_true",
                        default=False, help="Relabel reward or not")
    args = vars(parser.parse_args())
    print("Args: ", args)

    ENV = args['env']
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
    dpt = args['dpt']
    use_q_net = args['use_q_net']
    context_value = args['context_value']

    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }

    config.update({'dim': dim, 'rollin_type': 'uniform'})
 
    # test_env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    # print(test_env)
    seed = args['seed']
    if not args['fix_seed']:
        logger.info(f"no fix seed setting")
        num_envs = 20
        seed_list = [10000+i for i in range(num_envs)]
        env = make_minigrid_env(num_envs, seed_list)
    else:
        logger.info(f"set seed to {seed}")
        env = make_eval_env(seed=seed)

    assert args['cnn'], "Only support cnn models"

    if not args['fix_seed']:
        
        # Step1: eval_checkpoints_and_get_avg_return
        checkpoint_path = f"models/MiniGrid/{ENV}/noseed_cnn"
        checkpoint_dir = os.listdir(checkpoint_path)[0]
        checkpoint_path = f"{checkpoint_path}/{checkpoint_dir}"
        checkpoint_list = os.listdir(checkpoint_path)
        # checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt not in ["final.zip", "logs"]]
        checkpoint_version = [int(ckpt.split("_")[1]) for ckpt in checkpoint_list if ckpt not in ["final.zip", "logs", "eval_results.txt"]]
        checkpoint_version.sort()

        if "eval_results.txt" not in checkpoint_list:
            eval_results = []
            for v in checkpoint_version:
                model_path = f"{checkpoint_path}/ppo_{v}_steps.zip"
                model = PPO.load(model_path, env=env)
                # avg_reward = eval_models(model, env, seed_list)
                avg_reward = eval_models_multi_seed(model, env)
                logger.info(f"Avg reward {avg_reward} at {v} steps")
                eval_results.append(avg_reward)

            with open(f"{checkpoint_path}/eval_results.txt", "w") as f:
                eval_results_str = [str(v) for v in eval_results]
                f.write(", ".join(eval_results_str))

        else:
            with open(f"{checkpoint_path}/eval_results.txt", "r") as f:
                eval_results_str = f.read()
            eval_results = eval_results_str.split(", ")
            eval_results = [float(v) for v in eval_results]

        # print(eval_results)
        max_score = max(eval_results)
        upperline = 0.95 * max_score
        boardline = 0.05 * max_score
        for init_idx in range(len(eval_results)):
            if eval_results[init_idx] >= boardline:
                break
        for final_idx in range(len(eval_results)):
            if eval_results[final_idx] >= upperline:
                break
        print(init_idx, final_idx, len(eval_results))
        # collect 40 episode
        # if final_idx - init_idx + 1 >= 30: 
        more_init = False
        if more_init:
            samples = np.linspace(0, init_idx, 11)
            int_samples = np.round(samples).astype(int).tolist()[:-1]

            samples = np.linspace(init_idx, final_idx, 40)
            int_samples += np.round(samples).astype(int).tolist()
        else:
            samples = np.linspace(init_idx, final_idx, 30)
            int_samples = np.round(samples).astype(int).tolist()
        # else:
        #     int_samples = [i for i in range(init_idx, final_idx + 1)]
        
        # if len(eval_results) - final_idx >= 30:
        samples = np.linspace(final_idx, len(eval_results) - 1, 11)
        int_samples += np.round(samples).astype(int).tolist()[1:]

        logger.info(f"Collect {len(int_samples)} samples: {int_samples}")

        collect_train = args["train"]
        # 设置多进程
        if collect_train:
            if dpt:
                sample_env_num = 4000
                slices = 10
                num_per_slice = sample_env_num // slices
            else:
                sample_env_num = 10000
                slices = 5
                num_per_slice = sample_env_num // slices
            
            # 创建临时目录保存结果
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="minigrid_data_")
            logger.info(f"Created temporary directory for results: {temp_dir}")
            
            # 创建并启动进程
            processes = []
            for slice_idx in range(slices):
                mode = "dpt" if dpt else "ad"
                if context_value:
                    p = Process(
                        target=process_slice,
                        args=(slice_idx, slices, int_samples, checkpoint_version, temp_dir, checkpoint_path, num_per_slice, eval_results, mode, use_q_net)
                    )
                else:
                    p = Process(
                        target=process_slice,
                        args=(slice_idx, slices, int_samples, checkpoint_version, temp_dir, checkpoint_path, num_per_slice, None, mode, use_q_net)
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
                if dpt:
                    sample_res = {
                        "observations": np.concatenate([res['observations'] for res in sample_res_list], axis=0),
                        "actions": np.concatenate([res['actions'] for res in sample_res_list], axis=0),
                        "rewards": np.concatenate([res['rewards'] for res in sample_res_list], axis=0),
                        "dones": np.concatenate([res['dones'] for res in sample_res_list], axis=0),
                        "optimal_actions": np.concatenate([res['optimal_actions'] for res in sample_res_list], axis=0),
                    }
                elif use_q_net:
                    sample_res = {
                        "observations": np.concatenate([res['observations'] for res in sample_res_list], axis=0),
                        "actions": np.concatenate([res['actions'] for res in sample_res_list], axis=0),
                        "rewards": np.concatenate([res['rewards'] for res in sample_res_list], axis=0),
                        "dones": np.concatenate([res['dones'] for res in sample_res_list], axis=0),
                        "values": np.concatenate([res['values'] for res in sample_res_list], axis=0),
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
        else:
            sample_seed_list = [10000+i for i in range(40)]
            sample_env = make_minigrid_env(40, sample_seed_list)

            if dpt:
                sample_res = sample_dpt_traj_multi_seed(
                    int_samples, checkpoint_version, sample_env, num_envs=40, ckpt_path=checkpoint_path)
            elif use_q_net:
                sample_res = sample_traj_multi_seed(
                    int_samples, checkpoint_version, sample_env, num_envs=40, ckpt_path=checkpoint_path, use_q_net=True)
            else:
                sample_res = sample_traj_multi_seed(
                    int_samples, checkpoint_version, sample_env, num_envs=40, ckpt_path=checkpoint_path)
        if dpt:
            if not collect_train:
                save_path = f"datasets/MiniGrid/{ENV}/test_traj-dpt.pkl"
            else:
                save_path = f"datasets/MiniGrid/{ENV}/train_traj-dpt.pkl"
        else:
            if not collect_train:
                save_path = f"datasets/MiniGrid/{ENV}/test_traj.pkl"
            elif context_value:
                save_path = f"datasets/MiniGrid/{ENV}/train_traj-relabel.pkl"
            elif use_q_net:
                save_path = f"datasets/MiniGrid/{ENV}/train_traj-value.pkl"
            else:
                save_path = f"datasets/MiniGrid/{ENV}/train_traj-more.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(sample_res, f)

            



        

    
