import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model
from IPython import embed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from loguru import logger

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=1,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]


class ImageTransformer(Transformer):
    """Transformer class for Miniworld data."""

    def __init__(self, config):
        super().__init__(config)
        self.im_embd = 8

        size = self.config['image_size']

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        new_dim = self.im_embd + self.state_dim + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        query_images = x['query_images'][:, None, :]
        query_states = x['query_states'][:, None, :]
        context_images = x['context_images']
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        batch_size = query_states.shape[0]

        image_seq = torch.cat([query_images, context_images], dim=1)
        image_seq = image_seq.view(-1, *image_seq.size()[2:])

        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)

        context_states = torch.cat([query_states, context_states], dim=1)
        context_actions = torch.cat([
            torch.zeros(batch_size, 1, self.action_dim).to(device),
            context_actions,
        ], dim=1)
        context_rewards = torch.cat([
            torch.zeros(batch_size, 1, 1).to(device),
            context_rewards,
        ], dim=1)

        stacked_inputs = torch.cat([
            image_enc_seq,
            context_states,
            context_actions,
            context_rewards,
        ], dim=2)
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]
    

class DarkroomTransformer(nn.Module):
    """Transforemr class for Darkroom data."""

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
        self.im_embd = 64

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."

        config = GPT2Config(
            n_positions=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        new_dim = self.state_dim + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.mode = mode

    def forward(self, x):

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

        padding_action = torch.zeros(
            context_actions.shape[0], 1, self.action_dim, device=context_actions.device)
        padding_reward = torch.zeros(context_rewards.shape[0], 1, 1, device=context_rewards.device)
        if self.mode == "ad":
            context_actions = torch.cat(
                [padding_action, context_actions[:, :-1, :]], dim=1)
            context_rewards = torch.cat(
                [padding_reward, context_rewards[:, :-1, :]], dim=1)
        

        batch_size = context_states.shape[0]
    
        stacked_inputs = torch.cat([
            context_actions,
            context_rewards,
            context_states,
        ], dim=2)
        
        # 应用线性变换和层归一化
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        return preds


class DarkroomLambdaTransformer(nn.Module):
    """Transforemr class for Darkroom data."""

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
        self.im_embd = 64

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."

        config = GPT2Config(
            n_positions=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        new_dim = self.state_dim + self.action_dim + 2
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.mode = mode

    def forward(self, x):

        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']
        context_lambda = x['context_lambda']

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
        if len(context_lambda.shape) == 2:
            context_lambda = context_lambda[:, :, None]

        padding_action = torch.zeros(
            context_actions.shape[0], 1, self.action_dim, device=context_actions.device)
        padding_reward = torch.zeros(context_rewards.shape[0], 1, 1, device=context_rewards.device)
        if self.mode == "ad":
            context_actions = torch.cat(
                [padding_action, context_actions[:, :-1, :]], dim=1)
            context_rewards = torch.cat(
                [padding_reward, context_rewards[:, :-1, :]], dim=1)
            context_lambda = torch.cat(
                [padding_reward, context_lambda[:, :-1, :]], dim=1)
        
        batch_size = context_states.shape[0]
        # print(context_rewards.shape, context_lambda.shape)
        stacked_inputs = torch.cat([
            context_actions,
            context_rewards,
            context_lambda,
            context_states,
        ], dim=2)
        
        # 应用线性变换和层归一化
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        return preds
    



class MinigridTransformer(nn.Module):
    """Transformer class for Minigrid data."""

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
        self.im_embd = 64
        size = self.config['image_size']

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."

        # 修改GPT2Config中的n_positions参数，现在每个位置包含完整的(a,r,s')信息
        config = GPT2Config(
            n_positions=self.horizon,  # 序列长度与horizon相同
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        # 计算组合后的维度：图像编码 + one-hot动作 + 奖励
        new_dim = self.im_embd + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
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
            # 对于ad模式，next_state就是state序列向后移动一位
            context_states = x['context_states']

        
        # print(context_states)
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

        padding_action = torch.zeros(
            context_actions.shape[0], 1, self.action_dim, device=context_actions.device)
        padding_reward = torch.zeros(context_rewards.shape[0], 1, 1, device=context_rewards.device)
        if self.mode == "ad":
            context_actions = torch.cat(
                [padding_action, context_actions[:, :-1, :]], dim=1)
            context_rewards = torch.cat(
                [padding_reward, context_rewards[:, :-1, :]], dim=1)
        

        batch_size = context_states.shape[0]

        # 处理next_state图像序列
        image_seq = context_states
        image_seq = image_seq.view(-1, *image_seq.size()[2:])
        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)
    
        # 将next_state图像编码、动作和奖励在特征维度上拼接
        # 这样每个位置的pair是（action，reward，next_state）
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


class MinigridMultiheadTransformer(nn.Module):
    """Transformer class for Minigrid data."""

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
        self.im_embd = 64
        size = self.config['image_size']

        assert mode in ["ad", "dpt"], "You must select mode from ad or dpt."

        # 修改GPT2Config中的n_positions参数，现在每个位置包含完整的(a,r,s')信息
        config = GPT2Config(
            n_positions=self.horizon,  # 序列长度与horizon相同
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        # 计算组合后的维度：图像编码 + one-hot动作 + 奖励
        new_dim = self.im_embd + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.pred_values = nn.Linear(self.n_embd, 1)
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
            # 对于ad模式，next_state就是state序列向后移动一位
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

        padding_action = torch.zeros(
            context_actions.shape[0], 1, self.action_dim, device=context_actions.device)
        padding_reward = torch.zeros(context_rewards.shape[0], 1, 1, device=context_rewards.device)
        if self.mode == "ad":
            context_actions = torch.cat(
                [padding_action, context_actions[:, :-1, :]], dim=1)
            if auto_relabel:
                if pred_actions:
                    context_rewards = context_rewards
                else:
                    context_rewards = torch.cat(
                        [padding_reward, context_rewards[:, :-1, :]], dim=1)
            else:
                context_rewards = torch.cat(
                    [padding_reward, context_rewards[:, :-1, :]], dim=1)
        

        batch_size = context_states.shape[0]

        # 处理next_state图像序列
        image_seq = context_states
        
        image_seq = image_seq.view(-1, *image_seq.size()[2:])
        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)
    
        # 将next_state图像编码、动作和奖励在特征维度上拼接
        # 这样每个位置的pair是（action，reward，next_state）
        stacked_inputs = torch.cat([
            context_actions,
            context_rewards,
            image_enc_seq,
        ], dim=2)
        
        # 应用线性变换和层归一化
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        values = self.pred_values(transformer_outputs['last_hidden_state'])

        return preds, values
