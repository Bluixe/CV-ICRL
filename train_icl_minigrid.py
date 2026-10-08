import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
from IPython import embed

import matplotlib.pyplot as plt
import torch

import numpy as np
import common_args
import random
from dataset import Dataset, ImageDataset, MinigridDataset
from nets.net import MinigridTransformer, MinigridMultiheadTransformer
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_linear_bandit_data_filename,
    build_linear_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
    worker_init_fn,
)
from loguru import logger
from tqdm import tqdm

import wandb
os.environ.setdefault("WANDB_MODE", "disabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENV_LIST = [
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-SimpleCrossingS11N5-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-UnlockPickup-v0"
]

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.target_lr * (self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return self.optimizer.step()

def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

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
    parser.add_argument('--algorithm', type=str, default='ad')
    parser.add_argument('--use_finetune', action='store_true', default=False,
                        help='Use finetune data or not')
    parser.add_argument('--multi_env', action='store_true', default=False,
                        help='Use multiple environments or not')
    parser.add_argument('--context_value', action='store_true', default=False,
                        help='Relabel reward for training or not')
    parser.add_argument('--predict_value', action='store_true', default=False,
                        help='Automatically relabel reward for training or not')

    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
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
    context_value = args['context_value']
    predict_value = args['predict_value']

    gamma = 0.99
    expectile = 0.7
    tau = 0.005

    algorithm = args['algorithm']
    assert algorithm in ['ad', 'dpt']

    if algorithm in ['ad']:
        mode = 'ad'
    elif algorithm in ['dpt']:
        mode = 'dpt'
    
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
    
    if env.startswith('MiniGrid'):
        state_dim = 2
        action_dim = 7

        logger.info(f"Train in-context learning on {env}.")

        # path_train = f"datasets/MiniGrid/{env}/train_traj-finetune.pkl"
        if mode == "ad":
            if args['multi_env']:
                path_train = [f"datasets/MiniGrid/{e}/train_traj-more.pkl" for e in ENV_LIST]
            elif args['context_value'] or args['predict_value']:
                path_train = f"datasets/MiniGrid/{env}/train_traj-relabel.pkl"
            else:
                path_train = f"datasets/MiniGrid/{env}/train_traj-more.pkl"

            if args['multi_env']:
                path_test = [f"datasets/MiniGrid/{e}/test_traj.pkl" for e in ENV_LIST]
            else:
                path_test = f"datasets/MiniGrid/{env}/test_traj.pkl"

        elif mode == "dpt":
            path_train = f"datasets/MiniGrid/{env}/train_traj-dpt.pkl"
            path_test = f"datasets/MiniGrid/{env}/test_traj-dpt.pkl"
        
        filename = f"minigrid_{env}_model"

    else:
        raise NotImplementedError

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
        'store_gpu': True,
    }

    # logger.info(config)
    # exit(0)

    if env.startswith('MiniGrid'):
        logger.info("Using MiniGridTransformer model.")
        config.update({'image_size': 7, 'store_gpu': False})
        logger.info(config)
        if args["predict_value"]:
            model = MinigridMultiheadTransformer(config).to(device)
        else:
            model = MinigridTransformer(config).to(device)
    else:
        raise NotImplementedError

    params = {
        'batch_size': 64,
        'shuffle': True,
    }

    if env.startswith('MiniGrid'):
        params.update({'num_workers': 16, # 16
                'prefetch_factor': 2,
                'persistent_workers': True,
                'pin_memory': True,
                'batch_size': 32, # 32
                'worker_init_fn': worker_init_fn,
            })
        logger.info("Loading minigrid data...")
        if mode == "ad":
            train_dataset = MinigridDataset(path_train, config, mode)
            test_dataset = MinigridDataset(path_test, config, mode)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    test_loss = []
    train_loss = []

    logger.info("Num train batches: " + str(len(train_loader)))
    logger.info("Num test batches: " + str(len(test_loader)))

    wandb_name=f"{env}-{algorithm}-H{horizon}"
    wandb_group=f"{env}-{algorithm}"

    run = wandb.init(
        project="minigrid-icl",
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        )

    os.makedirs(f'models/MiniGrid/{env}/{algorithm}', exist_ok=True)

    for epoch in range(num_epochs):
        # EVALUATION
        logger.info(f"Epoch: {epoch + 1}")

        if mode == 'dpt':
            raise NotImplementedError
        else:
            start_time = time.time()
            if not predict_value:
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
                        batch_loss = loss.item() / horizon
                        epoch_test_loss += batch_loss
                        pbar.set_postfix({'loss': batch_loss})
                test_loss.append(epoch_test_loss / len(test_dataset))
                end_time = time.time()
                logger.info(f"Test loss: {test_loss[-1]}")
                logger.info(f"Test time: {end_time - start_time}")
            
            else:
                with torch.no_grad():
                    epoch_test_loss = 0.0
                    epoch_test_action_loss = 0.0
                    epoch_test_value_loss = 0.0
                    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test Epoch {epoch+1}")
                    for i, batch in pbar:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        true_actions = batch['context_actions']
                        if predict_value:
                            true_values = batch['context_rewards']
                        else:
                            true_values = batch['context_values']
                        pred_actions, pred_values = model(batch)
                        true_actions = true_actions.reshape(-1).long()
                        pred_actions = pred_actions.reshape(-1, action_dim)

                        loss_action = loss_fn(pred_actions, true_actions)
                        loss_value = torch.nn.functional.mse_loss(pred_values, true_values, reduction='sum')
                        loss = loss_action + loss_value
                        batch_loss = loss.item() / horizon
                        epoch_test_loss += batch_loss
                        epoch_test_action_loss += loss_action.item() / horizon
                        epoch_test_value_loss += loss_value.item() / horizon
                        pbar.set_postfix({'loss': batch_loss})
                test_loss.append(epoch_test_loss / len(test_dataset))
                # print(pred_actions)
                end_time = time.time()
                logger.info(f"Test loss: {test_loss[-1]}")
                logger.info(f"Test action loss: {epoch_test_action_loss / len(test_dataset)}")
                logger.info(f"Test value loss: {epoch_test_value_loss / len(test_dataset)}")
                logger.info(f"Test time: {end_time - start_time}")



        # TRAINING

        if mode == 'dpt':
            raise NotImplementedError
        else:
            epoch_train_loss = 0.0
            start_time = time.time()

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch+1}")
            for i, batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                if not predict_value:
                # if not use_value and not pred_reward and not sicql:
                    true_actions = batch['context_actions']
                    pred_actions = model(batch)
                    reward = batch['context_rewards']
                    # print(reward)

                    true_actions = true_actions.reshape(-1).long()
                    pred_actions = pred_actions.reshape(-1, action_dim)
                    pred_max_actions = torch.argmax(pred_actions, dim=-1)
                    pred_max_actions_one_hot = torch.nn.functional.one_hot(pred_max_actions, num_classes=action_dim).float()
                    num_of_actions = pred_max_actions_one_hot.sum(dim=0) / pred_max_actions_one_hot.shape[0]

                    optimizer.zero_grad()
                    loss = loss_fn(pred_actions, true_actions)
                        
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() / horizon
                    
                    epoch_train_loss += batch_loss
                    pbar.set_postfix({'loss': batch_loss})
                    
                    # 记录每个batch的loss到wandb
                    wandb.log({"batch_train_loss": batch_loss,
                            "action_0_prob": num_of_actions[0].item(),
                            "action_1_prob": num_of_actions[1].item(),
                            "action_2_prob": num_of_actions[2].item(),
                            "action_3_prob": num_of_actions[3].item(),
                            "action_4_prob": num_of_actions[4].item(),
                            "action_5_prob": num_of_actions[5].item(),
                            "action_6_prob": num_of_actions[6].item()})
                else:
                    true_actions = batch['context_actions']
                    if predict_value:
                        true_values = batch['context_rewards']
                    else:
                        true_values = batch['context_values']
                    pred_actions, pred_values = model(batch)

                    true_actions = true_actions.reshape(-1).long()
                    pred_actions = pred_actions.reshape(-1, action_dim)
                    pred_max_actions = torch.argmax(pred_actions, dim=-1)
                    pred_max_actions_one_hot = torch.nn.functional.one_hot(pred_max_actions, num_classes=action_dim).float()
                    num_of_actions = pred_max_actions_one_hot.sum(dim=0) / pred_max_actions_one_hot.shape[0]

                    optimizer.zero_grad()
                    loss_action = loss_fn(pred_actions, true_actions)
                    loss_value = torch.nn.functional.mse_loss(pred_values, true_values, reduction='sum')

                    loss = loss_action + loss_value
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() / horizon
                    batch_action_loss = loss_action.item() / horizon
                    batch_value_loss = loss_value.item() / horizon

                    epoch_train_loss += batch_loss
                    pbar.set_postfix({'loss': batch_loss})
                    
                    # 记录每个batch的loss到wandb
                    wandb.log({"batch_train_loss": batch_loss,
                            "batch_action_loss": batch_action_loss,
                            "batch_value_loss": batch_value_loss,
                            "action_0_prob": num_of_actions[0].item(),
                            "action_1_prob": num_of_actions[1].item(),
                            "action_2_prob": num_of_actions[2].item(),
                            "action_3_prob": num_of_actions[3].item(),
                            "action_4_prob": num_of_actions[4].item(),
                            "action_5_prob": num_of_actions[5].item(),
                            "action_6_prob": num_of_actions[6].item()})

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
            if (epoch + 1) % 5 == 0 or (env == 'linear_bandit' and (epoch + 1) % 10 == 0):
                
                if args['context_value'] and not args['multi_env']:
                    torch.save(model.state_dict(),
                            f'models/MiniGrid/{env}/{algorithm}/epoch{epoch+1}_context_value.pt')
                elif args['predict_value'] and not args['multi_env']:
                    torch.save(model.state_dict(),
                            f'models/MiniGrid/{env}/{algorithm}/epoch{epoch+1}_predict_value.pt')
                elif args['multi_env']:
                    if args['context_value']:
                        torch.save(model.state_dict(),
                                f'models/MiniGrid/{env}/{algorithm}/epoch{epoch+1}_multi_env_context_value.pt')
                    elif args['predict_value']:
                        torch.save(model.state_dict(),
                                f'models/MiniGrid/{env}/{algorithm}/epoch{epoch+1}_multi_env_predict_value.pt')
                    else:
                        torch.save(model.state_dict(),
                                f'models/MiniGrid/{env}/{algorithm}/epoch{epoch+1}_multi_env.pt')
                else:
                    torch.save(model.state_dict(), f'models/MiniGrid/{env}/{algorithm}/epoch{epoch+1}_more.pt')
        else:
            raise NotImplementedError

        # PLOTTING
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch: {epoch + 1}")
            logger.info(f"Test Loss:        {test_loss[-1]}")
            logger.info(f"Train Loss:       {train_loss[-1]}")
            logger.info("\n")

    if mode == "ad":
        if args['context_value'] and not args['multi_env']:
            torch.save(model.state_dict(), f'models/MiniGrid/{env}/{algorithm}/final_context_value.pt')
        elif args['predict_value'] and not args['multi_env']:
            torch.save(model.state_dict(), f'models/MiniGrid/{env}/{algorithm}/final_predict_value.pt')
        elif args['multi_env']:
            if args['context_value']:
                torch.save(model.state_dict(), f'models/MiniGrid/{env}/{algorithm}/final_multi_env_context_value.pt')
            elif args['predict_value']:
                torch.save(model.state_dict(), f'models/MiniGrid/{env}/{algorithm}/final_multi_env_predict_value.pt')
            else:
                torch.save(model.state_dict(), f'models/MiniGrid/{env}/{algorithm}/final_multi_env.pt')
        else:
            torch.save(model.state_dict(), f'models/MiniGrid/{env}/{algorithm}/final_more.pt')
    else:
        raise NotImplementedError
    print("Done.")
