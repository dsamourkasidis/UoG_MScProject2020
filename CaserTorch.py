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
    parser.add_argument('--num_filters', type=int, default=16,
                        help='Number of filters per filter size (default: 128)')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    return parser.parse_args()


class CaserTorch(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, device,num_filters,filter_sizes,
                 dropout_rate):
        super(CaserTorch, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.device = device
        self.filter_sizes=eval(filter_sizes)
        self.num_filters=num_filters
        self.dropout_rate=dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num+1,
            embedding_dim=self.hidden_size,
            # padding_idx=padding_idx
        )
        
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        
        #Horizontal Convolutional Layer
        
        self.horizontal_cnn = nn.ModuleList([nn.Conv2d(1,self.num_filters,(i,self.hidden_size))for i in self.filter_sizes])
        
        #Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1,self.num_filters,(self.state_size,1))
        
        #Fully Connected Layer
        self.vertical_dim = self.num_filters * self.hidden_size
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.vertical_dim + self.num_filters_total
        self.fc = nn.Linear(final_dim,item_num)
        
        #dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        


    def forward(self, inputs, inputs_lengths):
        input_emb = self.item_embeddings(inputs)
        mask = torch.ne(inputs, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        batch_size=inputs.size(0)
        embedded_chars_expanded = torch.reshape(input_emb,(batch_size,1,self.state_size,self.hidden_size))
        
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(embedded_chars_expanded))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out,h_out.size(2))
            pooled_outputs.append(p_out)
            
        h_pool = torch.cat(pooled_outputs,1)
        h_pool_flat = h_pool.view(-1,self.num_filters_total)
        
        v_out = nn.functional.relu(self.vertical_cnn(embedded_chars_expanded))
        v_flat = v_out.view(-1,self.vertical_dim)
        
        out = torch.cat([h_pool_flat,v_flat],1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
        
        
class CaserEvaluator(train_eval.Evaluator):
    def get_prediction(self, model, states, len_states, device):
        prediction = model(states.to(device).long(), len_states.to(device).long())
        return prediction


class CaserTrainer(train_eval.Trainer):

    def create_model(self):
        caserTorch = CaserTorch(hidden_size=self.args.hidden_factor, item_num=self.item_num,
                            state_size=self.state_size, device=self.device,num_filters=self.args.num_filters,
                            filter_sizes=self.args.filter_sizes,dropout_rate=self.args.dropout_rate)
        return caserTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        return out

    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        caser_evaluator = CaserEvaluator(device, args, data_directory, state_size, item_num)
        return caser_evaluator

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


TRAIN = 'train'
TEST = 'test'


def train_model(args, device, state_size, item_num):
    caser_trainer = CaserTrainer('caser_RC15', args, device, state_size, item_num)
    caser_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num):
    caserTorch = CaserTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device)
    checkpoint_handler = train_eval.CheckpointHandler('caser_RC15', device)
    optimizer = torch.optim.Adam(caserTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, caserTorch, optimizer)
    caserTorch.to(device)

    caser_evaluator = CaserEvaluator(device, args, data_directory, state_size, item_num)
    caser_evaluator.evaluate(caserTorch, 'test')


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