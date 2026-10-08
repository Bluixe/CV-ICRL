import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
import pickle

import torch
import numpy as np
import common_args
import random
from dataset import convert_to_tensor
from nets.net import device as default_device
from nets.impala_cnn import ImpalaCNN
from utils import worker_init_fn
from loguru import logger
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model
from gymnasium import spaces

import wandb
os.environ.setdefault("WANDB_MODE", "disabled")

# Procgen环境通常有15个动作
PROCGEN_ACTION_DIM = 15


class ProcgenTransformer(torch.nn.Module):
    """Transformer class for Procgen data using ImpalaCNN for image encoding."""

    def __init__(self, config, mode="ad"):
        super().__init__()
        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        # ImpalaCNN的emb_size
        self.im_embd = 256  # 与train_procgen_ppo_official.py中的emb_size保持一致

        # 图像尺寸 (C, H, W) - procgen默认是(3, 64, 64)
        self.image_shape = self.config.get('image_shape', (3, 64, 64))

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."

        # 创建GPT2配置
        gpt2_config = GPT2Config(
            n_positions=self.horizon,  # 序列长度与horizon相同
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt2_config)

        # 使用ImpalaCNN作为图像编码器（与train_procgen_ppo_official.py保持一致）
        # 创建observation_space用于ImpalaCNN
        observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.image_shape,
            dtype=np.uint8
        )
        self.image_encoder = ImpalaCNN(
            observation_space=observation_space,
            depths=[16, 32, 32],  # 与官方版本保持一致
            emb_size=self.im_embd
        )

        # 计算组合后的维度：图像编码 + one-hot动作 + 奖励
        new_dim = self.im_embd + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = torch.nn.LayerNorm(self.n_embd)
        self.pred_actions = torch.nn.Linear(self.n_embd, self.action_dim)
        self.mode = mode

    def forward(self, x, get_attentions=False):
        if self.mode == "dpt":
            context_states = x['context_states']
            optimal_actions = x['optimal_actions']
            if len(optimal_actions.shape) == 2:
                optimal_actions = optimal_actions.long()
            else:
                optimal_actions = optimal_actions.squeeze(2).long()
        else:
            # 对于ad模式
            context_states = x['context_states']

        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        # 确保action是long类型
        if len(context_actions.shape) == 2:
            context_actions = context_actions.long()
        else:
            context_actions = context_actions.squeeze(2).long()

        # 转换为one-hot编码
        context_actions = torch.nn.functional.one_hot(
            context_actions, num_classes=self.action_dim).float()

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        device = next(self.parameters()).device
        padding_action = torch.zeros(
            context_actions.shape[0], 1, self.action_dim).to(device)
        padding_reward = torch.zeros(context_rewards.shape[0], 1, 1).to(device)

        if self.mode == "ad":
            context_actions = torch.cat(
                [padding_action, context_actions[:, :-1, :]], dim=1)
            context_rewards = torch.cat(
                [padding_reward, context_rewards[:, :-1, :]], dim=1)

        batch_size = context_states.shape[0]

        # 处理图像序列
        # context_states shape: (batch_size, horizon, C, H, W)
        image_seq = context_states
        image_seq = image_seq.view(-1, *image_seq.size()[2:])  # (batch_size * horizon, C, H, W)

        # 归一化图像到[0, 1]范围（如果输入是uint8）
        if image_seq.dtype == torch.uint8:
            image_seq = image_seq.float() / 255.0

        # 使用ImpalaCNN编码图像
        image_enc_seq = self.image_encoder(image_seq)  # (batch_size * horizon, im_embd)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)  # (batch_size, horizon, im_embd)

        # 将图像编码、动作和奖励在特征维度上拼接
        stacked_inputs = torch.cat([
            context_actions,
            context_rewards,
            image_enc_seq,
        ], dim=2)

        # 应用线性变换和层归一化
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            output_attentions=get_attentions)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if get_attentions:
            return preds, transformer_outputs['attentions']
        else:
            return preds


class ProcgenMultiheadTransformer(torch.nn.Module):
    """Multihead Transformer class for Procgen data with value prediction head."""

    def __init__(self, config, mode="ad"):
        super().__init__()
        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        # ImpalaCNN的emb_size
        self.im_embd = 256  # 与train_procgen_ppo_official.py中的emb_size保持一致

        # 图像尺寸 (C, H, W) - procgen默认是(3, 64, 64)
        self.image_shape = self.config.get('image_shape', (3, 64, 64))

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."

        # 创建GPT2配置
        gpt2_config = GPT2Config(
            n_positions=self.horizon,  # 序列长度与horizon相同
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt2_config)

        # 使用ImpalaCNN作为图像编码器（与train_procgen_ppo_official.py保持一致）
        # 创建observation_space用于ImpalaCNN
        observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.image_shape,
            dtype=np.uint8
        )
        self.image_encoder = ImpalaCNN(
            observation_space=observation_space,
            depths=[16, 32, 32],  # 与官方版本保持一致
            emb_size=self.im_embd
        )

        # 计算组合后的维度：图像编码 + one-hot动作 + 奖励 + value
        # 当auto_relabel时，需要将context_values也作为输入
        new_dim = self.im_embd + self.action_dim + 1 + 1  # +1 for reward, +1 for value
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = torch.nn.LayerNorm(self.n_embd)
        self.pred_actions = torch.nn.Linear(self.n_embd, self.action_dim)
        self.pred_values = torch.nn.Linear(self.n_embd, 1)  # 添加value预测头
        self.mode = mode

    def forward(self, x, auto_relabel=False, pred_actions=False):
        if self.mode == "dpt":
            context_states = x['context_states']
            optimal_actions = x['optimal_actions']
            if len(optimal_actions.shape) == 2:
                optimal_actions = optimal_actions.long()
            else:
                optimal_actions = optimal_actions.squeeze(2).long()
        else:
            # 对于ad模式
            context_states = x['context_states']

        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        # 获取context_values（如果存在）
        context_values = None
        if 'context_values' in x:
            context_values = x['context_values']

        # 确保action是long类型
        if len(context_actions.shape) == 2:
            context_actions = context_actions.long()
        else:
            context_actions = context_actions.squeeze(2).long()

        # 转换为one-hot编码
        context_actions = torch.nn.functional.one_hot(
            context_actions, num_classes=self.action_dim).float()

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        # 处理context_values
        if context_values is not None:
            if len(context_values.shape) == 2:
                context_values = context_values[:, :, None]
        else:
            # 如果没有提供context_values，使用零填充
            device = next(self.parameters()).device
            context_values = torch.zeros_like(context_rewards).to(device)

        device = next(self.parameters()).device
        padding_action = torch.zeros(
            context_actions.shape[0], 1, self.action_dim).to(device)
        padding_reward = torch.zeros(context_rewards.shape[0], 1, 1).to(device)
        padding_value = torch.zeros(context_values.shape[0], 1, 1).to(device)

        if self.mode == "ad":
            context_actions = torch.cat(
                [padding_action, context_actions[:, :-1, :]], dim=1)
            if auto_relabel:
                if pred_actions:
                    # 预测actions时，使用真实的rewards和values
                    context_rewards = context_rewards
                    context_values = context_values
                else:
                    # 预测values时，使用shifted的rewards和values
                    context_rewards = torch.cat(
                        [padding_reward, context_rewards[:, :-1, :]], dim=1)
                    context_values = torch.cat(
                        [padding_value, context_values[:, :-1, :]], dim=1)
            else:
                context_rewards = torch.cat(
                    [padding_reward, context_rewards[:, :-1, :]], dim=1)
                context_values = torch.cat(
                    [padding_value, context_values[:, :-1, :]], dim=1)

        batch_size = context_states.shape[0]

        # 处理图像序列
        # context_states shape: (batch_size, horizon, C, H, W)
        image_seq = context_states
        image_seq = image_seq.view(-1, *image_seq.size()[2:])  # (batch_size * horizon, C, H, W)

        # 归一化图像到[0, 1]范围（如果输入是uint8）
        if image_seq.dtype == torch.uint8:
            image_seq = image_seq.float() / 255.0

        # 使用ImpalaCNN编码图像
        image_enc_seq = self.image_encoder(image_seq)  # (batch_size * horizon, im_embd)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)  # (batch_size, horizon, im_embd)

        # 将图像编码、动作、奖励和value在特征维度上拼接
        stacked_inputs = torch.cat([
            context_actions,
            context_rewards,
            context_values,
            image_enc_seq,
        ], dim=2)

        # 应用线性变换和层归一化
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        values = self.pred_values(transformer_outputs['last_hidden_state'])

        return preds, values


class ProcgenDataset(torch.utils.data.Dataset):
    """Dataset class for Procgen."""

    def __init__(self, path, config, mode="ad", include=[], reward_mode=None, use_reward=True, sample_stride=1):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        assert config['store_gpu'] == False
        self.store_gpu = config['store_gpu']
        self.config = config
        self.reward_mode = reward_mode
        self.sample_stride = sample_stride  # 每隔几个位置采样一个起点

        logger.info("ProcgenDataset Init! reward_mode: {}, sample_stride: {}".format(reward_mode, sample_stride))

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."
        self.include = include
        self.config = config
        self.mode = mode

        def load_data(file_path):
            """加载数据，支持.npz和.pkl格式"""
            if file_path.endswith('.npz'):
                # 使用npz格式加载
                data = np.load(file_path)
                traj = {
                    'observations': data['observations'],
                    'actions': data['actions'],
                    'rewards': data['rewards'],
                    'dones': data.get('dones', None),  # dones可能不存在
                    'values': data.get('values', None)
                }
                data.close()
                return traj
            else:
                # 使用pickle格式加载
                return pickle.load(open(file_path, 'rb'))

        if type(path) is not list:
            traj = load_data(path)
        else:
            trajs = []
            for p in path:
                traj = load_data(p)
                trajs.append(traj)

        if type(path) is list:
            context_states = []
            context_actions = []
            context_rewards = []
            if "values" in include:
                context_values = []
            for traj in trajs:
                logger.info(f"Processing trajectory with {len(traj['observations'])} observations.")
                if traj['observations'].shape[0] > 80000:
                    max_len = 80000
                    context_states.append(traj['observations'][:max_len])
                    context_actions.append(traj['actions'][:max_len])
                    context_rewards.append(traj['rewards'][:max_len])
                    if "values" in include:
                        context_values.append(traj['values'][:max_len])
                    # free memory
                    del traj['observations'], traj['actions'], traj['rewards']
                    if "values" in include:
                        del traj['values']
                else:
                    context_states.append(traj['observations'])
                    context_actions.append(traj['actions'])
                    context_rewards.append(traj['rewards'])
                    if "values" in include:
                        context_values.append(traj['values'])

            context_actions = np.concatenate(context_actions, axis=0)
            context_states = np.concatenate(context_states, axis=0)
            context_rewards = np.concatenate(context_rewards, axis=0)
            if "values" in include:
                context_values = np.concatenate(context_values, axis=0)

            logger.info(f"Context states shape: {context_states.shape}")
            logger.info(f"Context actions shape: {context_actions.shape}")
            logger.info(f"Context rewards shape: {context_rewards.shape}")
        else:
            context_states = traj['observations']
            context_actions = traj['actions']
            context_rewards = traj['rewards']
            if "values" in include:
                context_values = traj['values']

        # 检查数据形状，判断是否使用新的数据格式 (num_envs, 2000, ...)
        # 新格式：context_states是 (num_envs, 2000, C, H, W)，context_actions/rewards是 (num_envs, 2000)
        self.use_new_format = False
        # 新格式的特征：数据是5维（states）或2维（actions/rewards），且第二个维度（seq_length）大于horizon
        if len(context_states.shape) == 5:  # (num_envs, seq_len, C, H, W)
            if context_states.shape[1] > self.horizon:  # seq_len > horizon
                # 同时检查actions和rewards是否也是2维且seq_length匹配
                if (len(context_actions.shape) == 2 and context_actions.shape[0] == context_states.shape[0] and
                    context_actions.shape[1] == context_states.shape[1]):
                    self.use_new_format = True
                    self.num_envs = context_states.shape[0]
                    self.seq_length = context_states.shape[1]
                    # 计算有效的起始位置数量（考虑stride）
                    self.max_start_pos_raw = self.seq_length - self.horizon  # 2000 - 400 = 1600
                    # 使用stride来减少采样点：每隔sample_stride个位置采样一个起点
                    self.max_start_pos = (self.max_start_pos_raw + self.sample_stride - 1) // self.sample_stride
                    logger.info(f"Using new data format: states={context_states.shape}, actions={context_actions.shape}, num_envs={self.num_envs}, seq_length={self.seq_length}, max_start_pos_raw={self.max_start_pos_raw}, sample_stride={self.sample_stride}, max_start_pos={self.max_start_pos}")

        if not self.use_new_format:
            # 旧格式处理
            if len(context_rewards.shape) < 3:
                context_rewards = context_rewards[:, :, None]
            if len(context_actions.shape) < 3:
                context_actions = context_actions[:, :, None]
            if "values" in include:
                if len(context_values.shape) < 3:
                    context_values = context_values[:, :, None]

            if not use_reward:
                context_rewards = np.zeros_like(context_rewards)

            if "values" in include:
                self.dataset = {
                    'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
                    'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
                    'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
                    'context_values': convert_to_tensor(context_values, store_gpu=self.store_gpu),
                }
            else:
                self.dataset = {
                    'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
                    'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
                    'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
                }
        else:
            # 新格式：直接存储原始数据，在__getitem__中动态采样
            if not use_reward:
                context_rewards = np.zeros_like(context_rewards)

            # 确保数据维度正确
            if len(context_states.shape) == 4:  # (num_envs, seq_len, C, H, W)
                pass  # 已经是正确格式
            elif len(context_states.shape) == 5:  # 可能需要调整
                pass

            # 转换为tensor但不reshape，保持 (num_envs, seq_len, ...) 格式
            self.dataset = {
                'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
                'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
                'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
            }
            if "values" in include:
                self.dataset['context_values'] = convert_to_tensor(context_values, store_gpu=self.store_gpu)

    def __len__(self):
        if self.use_new_format:
            # 总共有 num_envs * max_start_pos 个可能的样本
            return self.num_envs * self.max_start_pos
        else:
            return len(self.dataset['context_states'])

    def __getitem__(self, index):
        'Generates one sample of data'

        if self.use_new_format:
            # 新格式：从 (num_envs, seq_len, ...) 中随机采样
            # index % (num_envs * max_start_pos) 来确定使用哪个样本
            total_samples = self.num_envs * self.max_start_pos
            index = index % total_samples
            env_idx = index // self.max_start_pos
            pos_idx_sampled = index % self.max_start_pos  # 采样后的位置索引（0到max_start_pos-1）
            # 转换为实际的起始位置：pos_idx_sampled * sample_stride
            pos_idx = pos_idx_sampled * self.sample_stride

            # 从选定的环境和位置提取horizon长度的数据
            context_states = self.dataset['context_states'][env_idx, pos_idx:pos_idx+self.horizon]
            context_actions = self.dataset['context_actions'][env_idx, pos_idx:pos_idx+self.horizon]
            context_rewards = self.dataset['context_rewards'][env_idx, pos_idx:pos_idx+self.horizon]

            # 确保维度正确
            if len(context_states.shape) == 3:  # (horizon, C, H, W)
                pass
            elif len(context_states.shape) == 4:  # 可能需要添加维度
                pass

            if len(context_actions.shape) == 1:  # (horizon,)
                context_actions = context_actions[:, None]
            if len(context_rewards.shape) == 1:  # (horizon,)
                context_rewards = context_rewards[:, None]

            if "values" in self.include:
                context_values = self.dataset['context_values'][env_idx, pos_idx:pos_idx+self.horizon]
                if len(context_values.shape) == 1:
                    context_values = context_values[:, None]
                res = {
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards,
                    'context_values': context_values,
                }
            else:
                res = {
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards,
                }
        elif "values" in self.include:
            res = {
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_rewards': self.dataset['context_rewards'][index],
                'context_values': self.dataset['context_values'][index],
            }
        elif self.reward_mode is not None:
            context_states = self.dataset['context_states'][index]
            context_actions = self.dataset['context_actions'][index]
            context_rewards = self.dataset['context_rewards'][index]
            # calculate the reward signals
            if self.reward_mode == 0:
                ## compute the remaining reward, i.e. the sum of rewards from the current step to the end of the trajectory
                context_values = torch.zeros_like(context_rewards)
                for i in range(self.horizon):
                    context_values[i] = context_rewards[i:].sum()
            elif self.reward_mode == 1:
                ## compute the remaining average reward
                context_values = torch.zeros_like(context_rewards)
                for i in range(self.horizon):
                    context_values[i] = context_rewards[i:].mean() * self.horizon
            elif self.reward_mode == 2:
                ## compute the remaining average reward
                context_values = torch.zeros_like(context_rewards)
                context_values[0] = context_rewards.mean() * self.horizon
                for i in range(1, self.horizon):
                    context_values[i] = max(context_rewards[i:].mean() * self.horizon, context_values[i-1])
            res = {
                'context_states': context_states,
                'context_actions': context_actions,
                'context_rewards': context_rewards,
                'context_values': context_values,
            }
        else:
            res = {
                'context_states': self.dataset['context_states'][index],
                'context_actions': self.dataset['context_actions'][index],
                'context_rewards': self.dataset['context_rewards'][index],
            }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
            if self.mode == "dpt":
                res['context_next_states'] = res['context_next_states'][perm]
            if "values" in self.include:
                res['context_values'] = res['context_values'][perm]

        return res


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--algorithm', choices=['ad'], default='ad')
    parser.add_argument('--use_finetune', action='store_true', default=False,
                        help='Use finetune data or not')
    parser.add_argument('--multi_env', action='store_true', default=False,
                        help='Use multiple environments or not')
    parser.add_argument('--use_value', action='store_true', default=False,
                        help='Use value function or not')
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
                        help='Automatically relabel reward for training or not')
    parser.add_argument('--use_epsilon', action='store_true', default=False,
                        help='Use epsilon-greedy data for training or not')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help='Device to use (cuda:0, cuda:1, or cpu). If not specified, uses cuda if available else cpu')
    parser.add_argument('--sample_stride', type=int, default=4,
                        help='Sample stride for new data format:每隔几个位置采样一个起点 (default: 4, reduces samples from 1.28M to 320K)')
    parser.add_argument('--max_batches_per_epoch', type=int, default=None,
                        help='Maximum number of batches to process per epoch (default: None, process all batches). Use this to reduce overfitting by limiting training data per epoch.')

    parser.add_argument('--train-data')
    parser.add_argument('--val-data')
    parser.add_argument('--output-dir')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    args = vars(parser.parse_args())
    if args['multi_env']:
        parser.error('The public release supports single-task Procgen AD/CV only.')
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = PROCGEN_ACTION_DIM  # Procgen通常有15个动作
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    use_value = args['use_value']
    adjust_loss = args['adjust_loss']
    reward_gain = args['reward_gain']
    suffix_reward = args['suffix_reward']
    pred_reward = args['pred_reward']
    reward_mode = args['reward_mode']
    wo_reward = args['wo_reward']
    relabel_reward = args['relabel_reward']
    auto_relabel = args['auto_relabel']
    sample_stride = args['sample_stride']
    max_batches_per_epoch = args['max_batches_per_epoch']

    algorithm = args['algorithm']
    assert algorithm in ['ad', 'dpt']

    if algorithm in ['ad']:
        mode = 'ad'
    elif algorithm in ['dpt']:
        mode = 'dpt'

    # 设置设备
    if args['device'] is not None:
        device = torch.device(args['device'])
    else:
        device = default_device
    logger.info(f"Using device: {device}")

    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }

    if env.startswith('procgen') or env in ['coinrun', 'bigfish', 'bossfight', 'caveflyer', 'chaser',
                                             'climber', 'dodgeball', 'fruitbot', 'heist', 'jumper',
                                             'leaper', 'maze', 'miner', 'ninja', 'plunder', 'starpilot']:
        # 提取环境名称（如果格式是procgen:coinrun，则提取coinrun）
        if ':' in env:
            env_name = env.split(':')[1]
        else:
            env_name = env

        state_dim = 2  # 占位符，实际使用图像
        action_dim = PROCGEN_ACTION_DIM

        logger.info(f"Train in-context learning on Procgen {env_name}.")

        if mode == "ad":
            if args['use_finetune']:
                path_train = [f"datasets/Procgen/{env_name}/train_traj-more.npz",
                             f"datasets/Procgen/{env_name}/train_traj-finetune.pkl"]
            elif args['multi_env']:
                # 如果支持多环境，可以在这里添加
                path_train = f"datasets/Procgen/{env_name}/train_traj-more.pkl"
            elif args['use_value']:
                path_train = f"datasets/Procgen/{env_name}/train_traj-value.pkl"
            elif args['relabel_reward'] or args['auto_relabel']:
                path_train = f"datasets/Procgen/{env_name}/train_traj-relabel-new.npz"
            elif args['use_epsilon']:
                path_train = f"datasets/Procgen/{env_name}/train_traj-epsilon.npz"
            else:
                path_train = f"datasets/Procgen/{env_name}/train_traj-official.npz"

            if args['multi_env']:
                path_test = f"datasets/Procgen/{env_name}/test_traj.pkl"
            elif args['auto_relabel']:
                path_test = f"datasets/Procgen/{env_name}/test_traj-relabel-new.npz"
            else:
                path_test = f"datasets/Procgen/{env_name}/test_traj-official.npz"

        elif mode == "dpt":
            path_train = f"datasets/Procgen/{env_name}/train_traj-dpt.pkl"
            path_test = f"datasets/Procgen/{env_name}/test_traj-dpt.pkl"

        filename = f"procgen_{env_name}_model"

    else:
        raise NotImplementedError(f"Environment {env} not supported for Procgen ICL training")

    path_train = args['train_data'] or path_train
    path_test = args['val_data'] or path_test
    output_dir = args['output_dir'] or f'models/Procgen/{env_name}/{algorithm}'
    if args['output_dir'] and os.path.exists(output_dir):
        raise FileExistsError(output_dir)
    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': False,  # Procgen图像较大，不存储在GPU上
        'image_shape': (3, 64, 64),  # Procgen RGB图像尺寸 (C, H, W)
    }

    # 根据模式选择模型
    if use_value or pred_reward or auto_relabel:
        logger.info("Using ProcgenMultiheadTransformer model.")
        model = ProcgenMultiheadTransformer(config, mode=mode).to(device)
    else:
        logger.info("Using ProcgenTransformer model.")
        model = ProcgenTransformer(config, mode=mode).to(device)
    logger.info(config)

    # params = {
    #     'batch_size': 32,  # Procgen图像较大，使用较小的batch size
    #     'shuffle': True,
    #     'num_workers': 16,
    #     'prefetch_factor': 2,
    #     'persistent_workers': True,
    #     'pin_memory': True,
    #     'worker_init_fn': worker_init_fn,
    # }
    params = {
        'batch_size': args['batch_size'],  # 从32降低到8（或更小，如4）
        'shuffle': True,
        'num_workers': args['num_workers'],  # 从16降低到4
        'prefetch_factor': 1,  # 从2降低到1
        'persistent_workers': True,
        'pin_memory': False,  # 改为False以节省显存
        'worker_init_fn': worker_init_fn,
    }

    if args['num_workers'] == 0:
        params.pop('prefetch_factor', None)
        params.pop('persistent_workers', None)
    logger.info("Loading procgen data...")
    if mode == "ad":
        if use_value:
            train_dataset = ProcgenDataset(path_train, config, mode, ['values'], sample_stride=sample_stride)
            test_dataset = ProcgenDataset(path_test, config, mode, ['values'], sample_stride=sample_stride)
        elif pred_reward:
            train_dataset = ProcgenDataset(path_train, config, mode, reward_mode=reward_mode, sample_stride=sample_stride)
            test_dataset = ProcgenDataset(path_test, config, mode, reward_mode=reward_mode, sample_stride=sample_stride)
        elif wo_reward:
            train_dataset = ProcgenDataset(path_train, config, mode, use_reward=False, sample_stride=sample_stride)
            test_dataset = ProcgenDataset(path_test, config, mode, use_reward=False, sample_stride=sample_stride)
        elif auto_relabel:
            train_dataset = ProcgenDataset(path_train, config, mode, ['values'], sample_stride=sample_stride)
            test_dataset = ProcgenDataset(path_test, config, mode, ['values'], sample_stride=sample_stride)
        else:
            train_dataset = ProcgenDataset(path_train, config, mode, sample_stride=sample_stride)
            test_dataset = ProcgenDataset(path_test, config, mode, sample_stride=sample_stride)
    else:
        raise NotImplementedError("DPT mode for Procgen is not implemented yet")

    logger.info("Done loading procgen data")

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if adjust_loss or suffix_reward:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    test_loss = []
    train_loss = []

    logger.info("Num train batches: " + str(len(train_loader)))
    logger.info("Num test batches: " + str(len(test_loader)))

    wandb_name = f"{env_name}-{algorithm}-H{horizon}"
    wandb_group = f"{env_name}-{algorithm}"

    run = wandb.init(
        project="procgen-icl",
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # EVALUATION
        logger.info(f"Epoch: {epoch + 1}")

        start_time = time.time()
        if not use_value and not pred_reward and not auto_relabel:
            with torch.no_grad():
                epoch_test_loss = 0.0
                epoch_test_original_loss = 0.0
                pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test Epoch {epoch+1}")
                for i, batch in pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch['context_actions']
                    pred_actions = model(batch)
                    true_actions = true_actions.reshape(-1).long()
                    pred_actions = pred_actions.reshape(-1, action_dim)

                    loss = loss_fn(pred_actions, true_actions)
                    if suffix_reward:
                        context_rewards = batch['context_rewards']
                        suffix_sum = context_rewards.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                        avg_suffix_reward = suffix_sum / (torch.arange(horizon, 0, -1, device=suffix_sum.device).float().view(1, -1, 1))
                        avg_reward = context_rewards.mean(dim=1).view(-1, 1, 1)
                        reward_gain = avg_suffix_reward - avg_reward
                        reward_gain = reward_gain.reshape(-1)
                        original_loss = torch.sum(loss)
                        loss = torch.sum(loss * torch.exp(reward_gain))

                    batch_loss = loss.item() / horizon
                    if suffix_reward:
                        epoch_test_original_loss += original_loss.item() / horizon
                    epoch_test_loss += batch_loss
                    pbar.set_postfix({'loss': batch_loss})
            test_loss.append(epoch_test_loss / len(test_dataset))
            end_time = time.time()
            if suffix_reward:
                logger.info(f"Original loss: {epoch_test_original_loss / len(test_dataset)}")
            logger.info(f"Test loss: {test_loss[-1]}")
            logger.info(f"Test time: {end_time - start_time}")
        else:
            with torch.no_grad():
                epoch_test_loss = 0.0
                epoch_test_action_loss = 0.0
                epoch_test_adjust_loss = 0.0
                epoch_test_value_loss = 0.0
                pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test Epoch {epoch+1}")
                for i, batch in pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch['context_actions']
                    true_values = batch['context_values']
                    pred_actions, pred_values = model(batch)
                    true_actions = true_actions.reshape(-1).long()
                    pred_actions = pred_actions.reshape(-1, action_dim)
                    pred_values = pred_values.reshape(-1, 1)
                    true_values = true_values.reshape(-1, 1)

                    loss_action = loss_fn(pred_actions, true_actions)
                    loss_value = torch.nn.functional.mse_loss(pred_values, true_values, reduction='sum')
                    batch_loss = (loss_action.item() + loss_value.item()) / horizon
                    epoch_test_loss += batch_loss
                    epoch_test_action_loss += torch.sum(loss_action).item() / horizon
                    epoch_test_value_loss += loss_value.item() / horizon
                    pbar.set_postfix({'loss': batch_loss})
            test_loss.append(epoch_test_loss / len(test_dataset))
            end_time = time.time()
            logger.info(f"Test loss: {test_loss[-1]}")
            logger.info(f"Test action loss: {epoch_test_action_loss / len(test_dataset)}")
            if auto_relabel or use_value or pred_reward:
                logger.info(f"Test value loss: {epoch_test_value_loss / len(test_dataset)}")
            logger.info(f"Test time: {end_time - start_time}")

        # TRAINING
        epoch_train_loss = 0.0
        start_time = time.time()

        # 限制每个epoch的batch数量，防止过拟合
        total_batches = len(train_loader)
        if max_batches_per_epoch is not None and max_batches_per_epoch > 0:
            batches_to_process = min(max_batches_per_epoch, total_batches)
            logger.info(f"Limiting training to {batches_to_process}/{total_batches} batches per epoch to reduce overfitting")
        else:
            batches_to_process = total_batches

        pbar = tqdm(enumerate(train_loader), total=batches_to_process, desc=f"Train Epoch {epoch+1}")
        actual_batches_processed = 0
        for i, batch in pbar:
            # 如果达到最大batch数量，提前停止
            if max_batches_per_epoch is not None and i >= max_batches_per_epoch:
                break
            actual_batches_processed += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            if not use_value and not pred_reward and not auto_relabel:
                true_actions = batch['context_actions']
                pred_actions = model(batch)
                reward = batch['context_rewards']

                true_actions = true_actions.reshape(-1).long()
                pred_actions = pred_actions.reshape(-1, action_dim)
                pred_max_actions = torch.argmax(pred_actions, dim=-1)
                pred_max_actions_one_hot = torch.nn.functional.one_hot(pred_max_actions, num_classes=action_dim).float()
                num_of_actions = pred_max_actions_one_hot.sum(dim=0) / pred_max_actions_one_hot.shape[0]

                optimizer.zero_grad()
                loss = loss_fn(pred_actions, true_actions)
                if suffix_reward:
                    context_rewards = batch['context_rewards']
                    suffix_sum = context_rewards.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                    avg_suffix_reward = suffix_sum / (torch.arange(horizon, 0, -1, device=suffix_sum.device).float().view(1, -1, 1))
                    avg_reward = context_rewards.mean(dim=1).view(-1, 1, 1)
                    reward_gain = avg_suffix_reward - avg_reward
                    reward_gain = reward_gain.reshape(-1)
                    original_loss = torch.sum(loss)
                    loss = torch.sum(loss * torch.exp(reward_gain))

                loss.backward()
                optimizer.step()
                batch_loss = loss.item() / horizon
                if suffix_reward:
                    batch_original_loss = original_loss.item() / horizon
                epoch_train_loss += batch_loss
                pbar.set_postfix({'loss': batch_loss})

                # 记录每个batch的loss到wandb
                log_dict = {
                    "batch_train_loss": batch_loss,
                    "action_0_prob": num_of_actions[0].item() if action_dim > 0 else 0,
                    "action_1_prob": num_of_actions[1].item() if action_dim > 1 else 0,
                    "action_2_prob": num_of_actions[2].item() if action_dim > 2 else 0,
                    "action_3_prob": num_of_actions[3].item() if action_dim > 3 else 0,
                    "action_4_prob": num_of_actions[4].item() if action_dim > 4 else 0,
                }
                if suffix_reward:
                    log_dict["batch_original_loss"] = batch_original_loss
                wandb.log(log_dict)
            else:
                true_actions = batch['context_actions']
                # if auto_relabel:
                #     true_values = batch['context_rewards']
                # else:
                true_values = batch['context_values']
                true_actions = true_actions.reshape(-1).long()

                if auto_relabel:
                    # # 首先训练 value 预测
                    # _, pred_values = model(batch, auto_relabel=True, pred_actions=False)
                    # optimizer.zero_grad()
                    # pred_values = pred_values.reshape(-1, 1)
                    # true_values = true_values.reshape(-1, 1)
                    # loss_value = torch.nn.functional.mse_loss(pred_values, true_values, reduction='sum')
                    # loss_value_item = loss_value.item()
                    # loss_value.backward()
                    # optimizer.step()

                    # # 然后训练 action 预测
                    # pred_actions, _ = model(batch, auto_relabel=True, pred_actions=True)
                    # pred_actions = pred_actions.reshape(-1, action_dim)
                    # pred_max_actions = torch.argmax(pred_actions, dim=-1)
                    # pred_max_actions_one_hot = torch.nn.functional.one_hot(pred_max_actions, num_classes=action_dim).float()
                    # num_of_actions = pred_max_actions_one_hot.sum(dim=0) / pred_max_actions_one_hot.shape[0]
                    # optimizer.zero_grad()
                    # loss_action = loss_fn(pred_actions, true_actions)
                    # loss_action_item = loss_action.item()
                    # loss_action.backward()
                    # optimizer.step()

                    pred_actions, pred_values = model(batch, auto_relabel=True, pred_actions=True)
                    pred_actions = pred_actions.reshape(-1, action_dim)
                    pred_values = pred_values.reshape(-1, 1)
                    true_values = true_values.reshape(-1, 1)
                    pred_max_actions = torch.argmax(pred_actions, dim=-1)
                    pred_max_actions_one_hot = torch.nn.functional.one_hot(pred_max_actions, num_classes=action_dim).float()
                    num_of_actions = pred_max_actions_one_hot.sum(dim=0) / pred_max_actions_one_hot.shape[0]
                    optimizer.zero_grad()
                    loss_action = loss_fn(pred_actions, true_actions)
                    loss_value = torch.nn.functional.mse_loss(pred_values, true_values, reduction='sum')
                    loss_action_item = loss_action.item()
                    loss_value_item = loss_value.item()
                    loss = loss_action + loss_value * 0.02
                    loss.backward()
                    optimizer.step()

                    loss_item = loss_action_item + loss_value_item
                    batch_loss = loss_item / horizon
                    batch_action_loss = loss_action_item / horizon
                    batch_value_loss = loss_value_item / horizon
                else:
                    # 对于use_value等模式，同时预测action和value
                    pred_actions, pred_values = model(batch)
                    pred_actions = pred_actions.reshape(-1, action_dim)
                    pred_values = pred_values.reshape(-1, 1)
                    true_values = true_values.reshape(-1, 1)
                    pred_max_actions = torch.argmax(pred_actions, dim=-1)
                    pred_max_actions_one_hot = torch.nn.functional.one_hot(pred_max_actions, num_classes=action_dim).float()
                    num_of_actions = pred_max_actions_one_hot.sum(dim=0) / pred_max_actions_one_hot.shape[0]

                    optimizer.zero_grad()
                    loss_action = loss_fn(pred_actions, true_actions)
                    loss_value = torch.nn.functional.mse_loss(pred_values, true_values, reduction='sum')
                    loss = loss_action + loss_value * 0.02
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() / horizon
                    batch_action_loss = torch.sum(loss_action).item() / horizon
                    batch_value_loss = loss_value.item() / horizon

                epoch_train_loss += batch_loss
                pbar.set_postfix({'loss': batch_loss})

                # 记录每个batch的loss到wandb
                log_dict = {
                    "batch_train_loss": batch_loss,
                    "batch_action_loss": batch_action_loss,
                    "batch_value_loss": batch_value_loss,
                    "action_0_prob": num_of_actions[0].item() if action_dim > 0 else 0,
                    "action_1_prob": num_of_actions[1].item() if action_dim > 1 else 0,
                    "action_2_prob": num_of_actions[2].item() if action_dim > 2 else 0,
                    "action_3_prob": num_of_actions[3].item() if action_dim > 3 else 0,
                    "action_4_prob": num_of_actions[4].item() if action_dim > 4 else 0,
                }
                wandb.log(log_dict)

        # 计算平均loss时，使用实际处理的batch数量
        # 注意：epoch_train_loss 是累加的 batch_loss，batch_loss 已经是每个样本的平均loss
        # 所以我们需要除以实际处理的batch数量，而不是数据集大小
        if max_batches_per_epoch is not None and batches_to_process < total_batches:
            # 如果限制了batch数量，使用实际处理的batch数量来计算平均loss
            if actual_batches_processed > 0:
                train_loss.append(epoch_train_loss / actual_batches_processed)
                logger.info(f"Processed {actual_batches_processed}/{total_batches} batches this epoch")
            else:
                train_loss.append(0.0)
                logger.warning("No batches processed this epoch!")
        else:
            # 使用数据集大小来计算平均loss（保持原有逻辑）
            train_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        logger.info(f"Train loss: {train_loss[-1]}")
        logger.info(f"Train time: {end_time - start_time}")

        # 记录每个epoch的loss到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss[-1] if train_loss else None,
            "test_loss": test_loss[-1] if test_loss else None,
            "train_time": end_time - start_time,
        })

        # LOGGING
        if mode == "ad":
            if (epoch + 1) % 1 == 0:
                if args['use_finetune']:
                    torch.save(model.state_dict(),
                            f'{output_dir}/epoch{epoch+1}_finetune_more.pt')
                elif args['wo_reward']:
                    torch.save(model.state_dict(),
                            f'{output_dir}/epoch{epoch+1}_wo_reward.pt')
                elif args['relabel_reward'] and not args['multi_env']:
                    torch.save(model.state_dict(),
                            f'{output_dir}/epoch{epoch+1}_relabel_reward.pt')
                elif args['auto_relabel'] and not args['multi_env']:
                    torch.save(model.state_dict(),
                            f'{output_dir}/epoch{epoch+1}_auto_relabel.pt')
                elif args['multi_env']:
                    if args['pred_reward']:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_multi_env_pred_reward_{reward_mode}.pt')
                    elif args['relabel_reward']:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_multi_env_relabel_reward.pt')
                    elif args['auto_relabel']:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_multi_env_auto_relabel.pt')
                    else:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_multi_env.pt')
                elif args['use_value']:
                    if args['adjust_loss']:
                        torch.save(model.state_dict(),
                            f'{output_dir}/epoch{epoch+1}_value_adjust.pt')
                    else:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_value.pt')
                elif args['pred_reward']:
                    torch.save(model.state_dict(),
                            f'{output_dir}/epoch{epoch+1}_pred_reward_{reward_mode}.pt')
                elif args['use_epsilon']:
                    torch.save(model.state_dict(), f'{output_dir}/epoch{epoch+1}_epsilon.pt')
                else:
                    if args['reward_gain']:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_reward_gain.pt')
                    elif args['suffix_reward']:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_suffix.pt')
                    else:
                        torch.save(model.state_dict(),
                                f'{output_dir}/epoch{epoch+1}_more.pt')

        # PLOTTING
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch: {epoch + 1}")
            logger.info(f"Test Loss:        {test_loss[-1]}")
            logger.info(f"Train Loss:       {train_loss[-1]}")
            logger.info("\n")

    if mode == "ad":
        if args['use_finetune']:
            torch.save(model.state_dict(), f'{output_dir}/final_finetune_more.pt')
        elif args['wo_reward']:
            torch.save(model.state_dict(), f'{output_dir}/final_wo_reward.pt')
        elif args['relabel_reward'] and not args['multi_env']:
            torch.save(model.state_dict(), f'{output_dir}/final_relabel_reward.pt')
        elif args['auto_relabel'] and not args['multi_env']:
            torch.save(model.state_dict(), f'{output_dir}/final_auto_relabel.pt')
        elif args['multi_env']:
            if args['reward_gain']:
                torch.save(model.state_dict(), f'{output_dir}/final_multi_env_reward_gain_{reward_mode}.pt')
            elif args['relabel_reward']:
                torch.save(model.state_dict(), f'{output_dir}/final_multi_env_relabel_reward.pt')
            elif args['auto_relabel']:
                torch.save(model.state_dict(), f'{output_dir}/final_multi_env_auto_relabel.pt')
            else:
                torch.save(model.state_dict(), f'{output_dir}/final_multi_env.pt')
        elif args['use_value']:
            if args['adjust_loss']:
                torch.save(model.state_dict(), f'{output_dir}/final_value_adjust.pt')
            else:
                torch.save(model.state_dict(), f'{output_dir}/final_value.pt')
        elif args['pred_reward']:
            torch.save(model.state_dict(), f'{output_dir}/final_pred_reward_{reward_mode}.pt')
        elif args['use_epsilon']:
            torch.save(model.state_dict(), f'{output_dir}/final_epsilon.pt')
        else:
            if args['reward_gain']:
                torch.save(model.state_dict(), f'{output_dir}/final_reward_gain.pt')
            elif args['suffix_reward']:
                torch.save(model.state_dict(), f'{output_dir}/final_suffix.pt')
            else:
                torch.save(model.state_dict(), f'{output_dir}/final_more.pt')

    print("Done.")
