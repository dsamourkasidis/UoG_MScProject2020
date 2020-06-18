import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utility import pad_history, calculate_hit


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
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
    def __init__(self, hidden_size, item_num, state_size, device, batch_size, gru_layers=1):
        super(GRUTorch, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num,
            embedding_dim=self.hidden_size,
            #padding_idx=padding_idx
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
        weight = next(self.parameters()).data
        hidden = weight.new_zeros([self.gru.num_layers, batch_size, self.hidden_size]).to(self.device)
        return hidden

    def forward(self, inputs, inputs_lengths):
        # reset the GRU hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden(self.batch_size)

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
        self.hidden = self.hidden.contiguous()
        self.hidden = self.hidden.view(-1, self.hidden.shape[2])

        # 4. run through actual linear layer
        output = self.fc(self.hidden)

        return output


def set_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def evaluate(grumodel, data_directory, args):
    topk = [5, 10, 15, 20]
    reward_click = args.r_click
    reward_buy = args.r_buy
    eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated=0
    total_clicks=0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]
    grumodel.eval()
    while evaluated<len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            id=eval_ids[evaluated]
            group=groups.get_group(id)
            history=[]
            for index, row in group.iterrows():
                state=list(history)
                len_states.append(state_size if len(state)>=state_size else 1 if len(state)==0 else len(state))
                state=pad_history(state,state_size,item_num)
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
        #prediction = grumodel(states.to(device).long(), len_state.to(device).long())
        #prediction = grumodelsess.run(GRUnet.output, feed_dict={GRUnet.inputs: states, GRUnet.len_state: len_states})
        sorted_list = np.argsort(prediction)
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward,
                      hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click=hit_clicks[i]/total_clicks
        hr_purchase=hit_purchase[i]/total_purchase
        ng_click=ndcg_clicks[i]/total_clicks
        ng_purchase=ndcg_purchase[i]/total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i],total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    print('#############################################################')


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


if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    device = set_device()
    data_directory = args.data

    state_size, item_num = get_stats(data_directory)

    train_loader = prepare_dataloader(data_directory)

    gruTorch = GRUTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device, batch_size=args.batch_size)
    gruTorch.to(device)
    gruTorch.train()
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gruTorch.parameters(), lr=args.lr)

    # Start training loop
    epoch_times = []
    total_step = 0
    for epoch in range(args.epoch):
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
            avg_loss += loss.item()

            total_step += 1
            if total_step % 200 == 0:
                #evaluate(gruTorch, data_directory, args)
                print("the loss in %dth batch is: %f" % (total_step, loss.item()))
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
            #if total_step % 2000 == 0:
                #evaluate(sess)
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, args.epoch, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
