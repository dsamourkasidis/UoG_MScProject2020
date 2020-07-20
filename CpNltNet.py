from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import argparse
from utility import set_device, calculate_simple_hit
import train_eval
import CpRec.residual_block as res_blk
import pandas as pd
import numpy as np
import os
import CpRec.BlockWiseEmbedding as block_emb

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--mode', default='test',
                        help='Train or test the model. "train" or "test"')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='../data/RC15',
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
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    return parser.parse_args()


class CpNltNet(nn.Module):
    def __init__(self, hidden_size, item_num, device, model_params):
        super(CpNltNet, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.model_params = model_params

        # --------------------
        # 1. Embeddings
#        self.item_embeddings = block_emb.BlockWiseEmbeddingForInput(item_num, self.hidden_size, device,
 #                                                                   self.emb_param['block'], self.emb_param['factor'])
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
            # padding_idx=padding_idx
        )
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        # ---------------------
        # 2. Residual blocks
        residual_blocks = []
        bl = res_blk.CpResidualBlockAdjacentBlock(self.model_params['dilations'][0], self.hidden_size, self.model_params['dilated_channels'],
                                               self.model_params['kernel_size'], 0, None)
        residual_blocks.append(bl)
        for layer_id, dil in enumerate(self.model_params['dilations'][1:]):
            bl = res_blk.CpResidualBlockAdjacentBlock(dil, self.hidden_size, self.model_params['dilated_channels'],
                                                   self.model_params['kernel_size'], layer_id+1, bl)
            residual_blocks.append(bl)
        self.res_dil_convs = nn.ModuleList(residual_blocks)

        # ---------------------
        # 3. FC
        self.fc = nn.Linear(self.hidden_size, item_num)

    def forward(self, inputs, inputs_lengths):
        # ---------------------
        # 1. embed the input
        context_embedding = self.item_embeddings(inputs)
        mask = (~inputs.eq(self.item_num)).float().unsqueeze(-1)
        context_embedding *= mask

        dilate_output = context_embedding
        for dilated in self.res_dil_convs:
            dilate_output = dilated(dilate_output)
            dilate_output *= mask

        state_hidden = self.extract_unpadded(dilate_output, inputs_lengths - 1)

        output = self.fc(state_hidden)
        return output

    def extract_unpadded(self, data, ind):
        """
        Get true elements from each sequence (not padded)
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """
        batch_range = torch.arange(0, data.shape[0], dtype=torch.int64).to(self.device)
        indices = torch.stack([batch_range, ind], dim=1)
        res = data[indices.transpose(0, 1).tolist()]
        return res


class CpNltNetEvaluator(train_eval.Evaluator):
    def get_prediction(self, model, states, len_states, device):
        prediction = model(states.to(device).long(), len_states.to(device).long())
        return prediction

    def create_val_loader(self, filepath):
        replay_buffer = pd.read_pickle(filepath)
        replay_buffer_dic = replay_buffer.to_dict()
        states = replay_buffer_dic['state'].values()
        states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
        len_states = replay_buffer_dic['len_state'].values()
        len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
        targets = replay_buffer_dic['action'].values()
        targets = torch.from_numpy(np.fromiter(targets, dtype=np.long)).long()
        train_data = TensorDataset(states, len_states, targets)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=128, drop_last=True)
        return train_loader

    def evaluate(self, model, val_or_test):
        topk = [5, 10, 15, 20]
        val_loader = self.create_val_loader('../data/RC15/replay_buffer_test.df')
        hit = [0, 0, 0, 0]
        ndcg = [0, 0, 0, 0]

        with torch.no_grad():
            model.eval()
            print('Evaluation started...')
            for batch_idx, (states, len_states, target) in enumerate(val_loader):
                prediction = self.get_prediction(model, states, len_states, self.device)
                del states
                del len_states
                torch.cuda.empty_cache()
                sorted_list = np.argsort(prediction.tolist())
                calculate_simple_hit(sorted_list, topk, target.tolist(), hit, ndcg)

                if batch_idx % 200 == 0:
                    print('Evaluated {} / {}'.format(batch_idx, len(val_loader)))
            print('#############################################################')

            val_acc = 0
            for i in range(len(topk)):
                hr = hit[i] / len(val_loader.dataset)
                ng = ndcg[i] / len(val_loader.dataset)
                val_acc = val_acc + hr + ng
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                print('hr ndcg @ %d : %f, %f' % (topk[i], hr, ng))

            print('#############################################################')
            if val_or_test == "val":
                return val_acc


class CpNltNetTrainer(train_eval.Trainer):

    def create_model(self, model_params):
        cpNltNetTorch = CpNltNet(hidden_size=self.args.hidden_factor, item_num=self.item_num,
                                 device=self.device, model_params=model_params)
        total_params = sum(p.numel() for p in cpNltNetTorch.parameters() if p.requires_grad)
        print(total_params)
        return cpNltNetTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        return out

    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        cpNlt_evaluator = CpNltNetEvaluator(device, args, data_directory, state_size, item_num)
        return cpNlt_evaluator

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


TRAIN = 'train'
TEST = 'test'


def train_model(args, device, state_size, item_num, model_name, model_param):
    nlt_trainer = CpNltNetTrainer(model_name, args, device, state_size, item_num, model_param)
    nlt_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num, model_name, model_params):
    cpNltTorch = CpNltNet(hidden_size=args.hidden_factor, item_num=item_num, device=device, model_params=model_params)
    checkpoint_handler = train_eval.CheckpointHandler(model_name, device)
    optimizer = torch.optim.Adam(cpNltTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, cpNltTorch, optimizer)
    cpNltTorch.to(device)

    cpNlt_evaluator = CpNltNetEvaluator(device, args, data_directory, state_size, item_num)
    cpNlt_evaluator.evaluate(cpNltTorch, 'test')


if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    device = set_device()
    data_directory = args.data
    model_name = 'CpNltNet_RC15_24layers_adjblk'
    model_param = {
        'dilated_channels': 64,  # larger is better until 512 or 1024
        'dilations': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # YOU should tune this hyper-parameter, refer to the paper.
        'kernel_size': 3
        # 'param-share' : ''
    }

    state_size, item_num = train_eval.get_stats(data_directory)
    train_loader = train_eval.prepare_dataloader(data_directory, args.batch_size)

    if args.mode.lower() == TRAIN:
        train_model(args, device, state_size, item_num, model_name, model_param)
    else:
        test_model(device, args, data_directory, state_size, item_num, model_name, model_param)


