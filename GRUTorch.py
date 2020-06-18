import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utility import pad_history, calculate_hit
from shutil import copyfile


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
    parser.add_argument('--resume', type=int, default=1,
                        help='flag for resume. 1: resume training; 0: train from start')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    return parser.parse_args()


class GRUTorch(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, device, gru_layers=1):
        super(GRUTorch, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num,
            embedding_dim=self.hidden_size,
            # padding_idx=padding_idx
        )
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, item_num)

    def init_hidden(self, batch_size):
        return torch.zeros(self.gru.num_layers, batch_size, self.hidden_size).to(self.device)


    def forward(self, inputs, inputs_lengths):
        # reset the GRU hidden state. Must be done before you run a new batch. Otherwise the GRU will treat
        # a new batch as a continuation of a sequence
        batch_size = inputs.size(0)
        self.hidden = self.init_hidden(batch_size)

        # ---------------------
        # 1. embed the input
        x = self.item_embeddings(inputs)

        # ---------------------
        # 2.
        # pack_padded_sequence so that padded items in the sequence won't be shown to the GRU
        x = torch.nn.utils.rnn.pack_padded_sequence(x, inputs_lengths, batch_first=True, enforce_sorted=False)

        # now run through GRU
        x, self.hidden = self.gru(x, self.hidden)

        # undo the packing operation
        #x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # ---------------------
        # 3. prepare to run through linear layer
        #self.hidden = self.hidden.contiguous()
        self.hidden = self.hidden.view(-1, self.hidden.shape[2])

        # 4. run through actual linear layer
        output = self.fc(self.hidden)
        # x = self.fc(x)
        return output


def set_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def save_checkpoint(state, is_best, best_filename):
    checkpoint = 'gru_checkpoint.pth.tar'
    print('saved checkpoint')
    torch.save(state, checkpoint)
    if is_best:
        copyfile(checkpoint, best_filename)
        print('saved best checkpoint')


def load_from_checkpoint(args, best_filename, gru_model, optimizer, device):
    if args.resume and os.path.isfile(best_filename):
        print("=> loading checkpoint")
        checkpoint = torch.load(best_filename)
        start_epoch = checkpoint['epoch']
        min_loss = checkpoint['min_loss']
        gru_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        for optim_state in optimizer.state.values():
            for k, v in optim_state.items():
                if torch.is_tensor(v):
                    optim_state[k] = v.to(device)
        print("=> loaded checkpoint. Epoch: {})".format(checkpoint['epoch']))
    else:
        print('no checkpoint at {}'.format(best_filename))
        start_epoch = 0
        min_loss = 1000
    return start_epoch, min_loss


def prepare_dataloader(data_directory):
    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
    targets = replay_buffer_dic['action'].values()
    targets = torch.from_numpy(np.fromiter(targets, dtype=np.long)).long()
    train_data = TensorDataset(states, len_states, targets)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)
    return train_loader


def get_stats(data_dir):
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    return state_size, item_num


def evaluate(grumodel, data_directory, args):
    topk = [5, 10, 15, 20]
    reward_click = args.r_click
    reward_buy = args.r_buy
    eval_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
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
    grumodel.eval()
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
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
        prediction = grumodel(states.to(device).long(), len_states.to(device).long())
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


if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    device = set_device()
    data_directory = args.data

    state_size, item_num = get_stats(data_directory)
    train_loader = prepare_dataloader(data_directory)

    gruTorch = GRUTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gruTorch.parameters(), lr=args.lr)

    checkpoint_file = 'gru__best.pth.tar'
    start_epoch, min_loss = load_from_checkpoint(args, checkpoint_file, gruTorch, optimizer, device)
    gruTorch.to(device)

    # Start training loop
    epoch_times = []
    total_step = 0
    for epoch in range(start_epoch, args.epochs):
        gruTorch.train()
        start_time = time.clock()
        avg_loss = 0.
        counter = 0
        for state, len_state, target in train_loader:
            counter += 1
            gruTorch.zero_grad()

            out = gruTorch(state.to(device).long(), len_state.to(device).long())
            loss = criterion(out, target.to(device).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_step += 1
            if total_step % 200 == 0:
                print("the loss in %dth batch is: %f" % (total_step, loss.item()))
                print("Epoch {}......Batch: {}/{}....... Loss: {}".format(epoch, counter,
                                                                          len(train_loader),
                                                                          loss.item()))
            if total_step % 2000 == 0:
                evaluate(gruTorch, data_directory, args)
                is_best = loss.item() < min_loss
                min_loss = min(min_loss, loss.item())
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': gruTorch.state_dict(),
                    'min_loss': min_loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best, checkpoint_file)
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, args.epochs, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
