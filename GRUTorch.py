import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


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
    def __init__(self, hidden_size, item_num, state_size, device, gru_layers=1):
        super(GRUTorch, self).__init__()
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
            num_layers=gru_layers
            #batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, item_num)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new_zeros([self.gru.num_layers, batch_size, self.hidden_size]).to(self.device)
        return hidden

    def forward(self, inputs, inputs_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        batch_size, seq_len = inputs.size()
        self.hidden = self.init_hidden(batch_size)

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        x = self.item_embeddings(inputs)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x = torch.nn.utils.rnn.pack_padded_sequence(x, inputs_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        x, self.hidden = self.gru(x, self.hidden)

        # undo the packing operation
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
#        x = x.contiguous()
 #       x = x.view(-1, x.shape[2])
        self.hidden = self.hidden.contiguous()
        self.hidden = self.hidden.view(-1, self.hidden.shape[2])

        # run through actual linear layer
  #      x = self.fc(x)
        self.hidden = self.fc(self.hidden)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        #X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        #x = x.view(batch_size, seq_len, self.hidden_size)

        #y_hat = x
        return self.hidden


def set_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    device = set_device()
    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk=[5,10,15,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

    gruTorch = GRUTorch(hidden_size=args.hidden_factor, item_num=item_num, state_size=state_size, device=device)
    gruTorch.to(device)
    gruTorch.train()

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    replay_buffer_dic = replay_buffer.to_dict()
    state = replay_buffer_dic['state'].values()
    state = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in state]).long()
    len_state = replay_buffer_dic['len_state'].values()
    len_state = torch.from_numpy(np.fromiter(len_state, dtype=np.long)).long()
    target = replay_buffer_dic['action'].values()
    target = torch.from_numpy(np.fromiter(target, dtype=np.long)).long()
    train_data = TensorDataset(state, len_state, target)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)

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
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            total_step += 1
            if total_step % 200 == 0:
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
