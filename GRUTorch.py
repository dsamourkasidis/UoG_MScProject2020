import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
import time
import numpy as np
from utility import pad_history, calculate_hit, set_device
from shutil import copyfile
import train_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--mode', default='train',
                        help='Train or test the model. "train" or "test"')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data/RC15',
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
            num_embeddings=item_num+1,
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


class GRUEvaluator(train_eval.Evaluator):
    def get_prediction(self, model, states, len_states, device):
        prediction = model(states.to(device).long(), len_states.to(device).long())
        return prediction


class GRUTrainer(train_eval.Trainer):

    def create_model(self):
        gruTorch = GRUTorch(hidden_size=self.args.hidden_factor, item_num=self.item_num,
                            state_size=self.state_size, device=self.device)
        return gruTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        return out

    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        gru_evaluator = GRUEvaluator(device, args, data_directory, state_size, item_num)
        return gru_evaluator

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


TRAIN = 'train'
TEST = 'test'


def train_model(args, device, state_size, item_num):
    gru_trainer = GRUTrainer('gru', args, device, state_size, item_num)
    gru_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num):
    gruTorch = GRUTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device)
    checkpoint_handler = train_eval.CheckpointHandler('gru', device)
    optimizer = torch.optim.Adam(gruTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, gruTorch, optimizer)
    gruTorch.to(device)

    gru_evaluator = GRUEvaluator(device, args, data_directory, state_size, item_num)
    gru_evaluator.evaluate(gruTorch, 'test')


if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    device = set_device()
    data_directory = args.data

    state_size, item_num = train_eval.get_stats(data_directory)
    train_loader = train_eval.prepare_dataloader(data_directory, args.batch_size)

    if args.mode.lower() == TRAIN:
        train_model(args, device, state_size, item_num)
    else:
        test_model(device, args, data_directory, state_size, item_num)


