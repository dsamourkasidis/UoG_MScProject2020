import pandas as pd
import os
from utility import pad_history, calculate_hit
import torch
import numpy as np
from abc import ABC, abstractmethod
from shutil import copyfile
import time
from torch.utils.data import TensorDataset, DataLoader


def prepare_dataloader(data_directory, batch_size):
    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
    targets = replay_buffer_dic['action'].values()
    targets = torch.from_numpy(np.fromiter(targets, dtype=np.long)).long()
    train_data = TensorDataset(states, len_states, targets)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def get_stats(data_directory):
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    return state_size, item_num


class Evaluator(ABC):
    def __init__(self, device, args, data_directory, state_size, item_num):
        self.device = device
        self.args = args
        self.data_directory = data_directory
        self.state_size = state_size
        self.item_num = item_num

    def evaluate(self, model):
        topk = [5, 10, 15, 20]
        reward_click = self.args.r_click
        reward_buy = self.args.r_buy
        eval_sessions = pd.read_pickle(os.path.join(self.data_directory, 'sampled_val.df'))
        eval_ids = eval_sessions.session_id.unique()
        groups = eval_sessions.groupby('session_id')
        batch = 100
        evaluated = 0
        total_clicks = 0.0
        total_purchase = 0.0
        total_reward = [0, 0, 0, 0]
        hit_clicks = [0, 0, 0, 0]
        ndcg_clicks = [0, 0, 0, 0]
        hit_purchase = [0, 0, 0, 0]
        ndcg_purchase = [0, 0, 0, 0]
        model.eval()
        while evaluated < len(eval_ids):
            states, len_states, actions, rewards = [], [], [], []
            for i in range(batch):
                id = eval_ids[evaluated]
                group = groups.get_group(id)
                history = []
                for index, row in group.iterrows():
                    state = list(history)
                    len_states.append(self.state_size if len(state) >= self.state_size else 1 if len(state) == 0 else len(state))
                    state = pad_history(state, self.state_size, self.item_num)
                    states.append(state)
                    action = row['item_id']
                    is_buy = row['is_buy']
                    reward = reward_buy if is_buy == 1 else reward_click
                    if is_buy == 1:
                        total_purchase += 1.0
                    else:
                        total_clicks += 1.0
                    actions.append(action)
                    rewards.append(reward)
                    history.append(row['item_id'])
                evaluated += 1
            states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
            len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
            prediction = self.get_prediction(model, states, len_states, self.device)
            sorted_list = np.argsort(prediction.tolist())
            calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward,
                          hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
        print('#############################################################')
        print('total clicks: %d, total purchases:%d' % (total_clicks, total_purchase))
        for i in range(len(topk)):
            hr_click = hit_clicks[i] / total_clicks
            hr_purchase = hit_purchase[i] / total_purchase
            ng_click = ndcg_clicks[i] / total_clicks
            ng_purchase = ndcg_purchase[i] / total_purchase
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
            print('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
            print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
        print('#############################################################')


    @abstractmethod
    def get_prediction(self, model, states, len_states, device):
        pass


class CheckpointHandler:
    def __init__(self, model_name, device):
        self.best_checkpoint = model_name + '_best.pth.tar'
        self.checkpoint = model_name + '_checkpoint.pth.tar'
        self.device = device

    def save_checkpoint(self, state, is_best):
        print('saved checkpoint')
        torch.save(state, self.checkpoint)
        if is_best:
            copyfile(self.checkpoint, self.best_checkpoint)
            print('saved best checkpoint')

    def load_from_checkpoint(self, resume, model, optimizer):
        if resume and os.path.isfile(self.best_checkpoint):
            print("=> loading checkpoint")
            checkpoint = torch.load(self.best_checkpoint)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            for optim_state in optimizer.state.values():
                for k, v in optim_state.items():
                    if torch.is_tensor(v):
                        optim_state[k] = v.to(self.device)
            print("=> loaded checkpoint. Epoch: {})".format(checkpoint['epoch']))
        else:
            print('no checkpoint at {}'.format(self.best_checkpoint))
            start_epoch = 0
            min_loss = 1000
        return start_epoch, min_loss


class Trainer(ABC):
    def __init__(self, model_name, args, device, state_size, item_num):
        self.args = args
        self.device = device
        self.model_name = model_name
        self.state_size = state_size
        self.item_num = item_num
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def get_model_out(self, state, len_state):
        pass

    @abstractmethod
    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def get_criterion(self):
        pass

    def train(self, train_loader):

        checkpoint_handler = CheckpointHandler(self.model_name, self.device)
        start_epoch, min_loss = checkpoint_handler.load_from_checkpoint(self.args.resume, self.model, self.optimizer)
        self.model.to(self.device)

        evaluator = self.get_evaluator(self.device, self.args, self.args.data, self.state_size, self.item_num)
        criterion = self.get_criterion()

        # Start training loop
        epoch_times = []
        total_step = 0
        for epoch in range(start_epoch, self.args.epochs):
            self.model.train()
            start_time = time.clock()
            avg_loss = 0.
            counter = 0
            for state, len_state, target in train_loader:
                counter += 1
                self.model.zero_grad()

                out = self.get_model_out(state.to(self.device).long(), len_state.to(self.device).long())
                loss = criterion(out, target.to(self.device).long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_step += 1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss.item()))
                    print("Epoch {}......Batch: {}/{}....... Loss: {}".format(epoch, counter,
                                                                              len(train_loader),
                                                                              loss.item()))
                if total_step % 200 == 0:
                    evaluator.evaluate(self.model)
                    self.model.train()
                    is_best = loss.item() < min_loss
                    min_loss = min(min_loss, loss.item())
                    checkpoint_handler.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'min_loss': min_loss,
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best)
            current_time = time.clock()
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, self.args.epochs, avg_loss / len(train_loader)))
            print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
            epoch_times.append(current_time - start_time)
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))