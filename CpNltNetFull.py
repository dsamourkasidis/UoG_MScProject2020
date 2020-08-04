import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import torch
import torch.nn as nn
import argparse
import train_eval
from utility import set_device
from CpRec import residual_block as res_blk
import pandas as pd
import numpy as np
import os
import logger
from torch.utils.data import TensorDataset, DataLoader
# import CpRec.BlockWiseEmbedding as block_emb


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--mode', default='train',
                        help='Train or test the model. "train" or "test"')
    parser.add_argument('--epochs', type=int, default=14,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data/ML100',
                        help='data directory')
    parser.add_argument('--resume', type=int, default=1,
                        help='flag for resume. 1: resume training; 0: train from start')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=512,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dilations', type=str, default="1,4,1,4",
                        help='dilations for res blocks. eg "1,4,1,4"')
    parser.add_argument('--modelname', type=str, default="CpNltNetFull_ML100_8layers_adjblk",
                        help='model name. eg for checkpoint filename')
    parser.add_argument('--resblocktype', type=int, default=4,
                        help='Residual block type. 0-Simple, 1-CROSSLAYER, 2-CROSSBLOCK, 3-ADJACENTLAYER, 4-ADJACENTBLOCK')
    return parser.parse_args()
    

class CpNltNetFull(nn.Module):
    def __init__(self, hidden_size, item_num, device, model_params):
        super(CpNltNetFull, self).__init__()
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
        # bl = res_blk.CpResidualBlockAdjacentBlock(self.model_params['dilations'][0], self.hidden_size, self.model_params['dilated_channels'],
        #                                           self.model_params['kernel_size'], 0, None)
        bl = self.create_residual(self.model_params['resblocktype'], self.model_params['dilations'][0], self.hidden_size, self.model_params['dilated_channels'],
                                                  self.model_params['kernel_size'], 0, None)
                                                  
        residual_blocks.append(bl)
        for layer_id, dil in enumerate(self.model_params['dilations'][1:]):
            # bl = res_blk.CpResidualBlockAdjacentBlock(dil, self.hidden_size, self.model_params['dilated_channels'],
            #                                        self.model_params['kernel_size'], layer_id+1, bl)
            bl = self.create_residual(self.model_params['resblocktype'], dil, self.hidden_size, self.model_params['dilated_channels'],
                                                  self.model_params['kernel_size'], layer_id+1, bl)
            residual_blocks.append(bl)
        self.res_dil_convs = nn.ModuleList(residual_blocks)

        # ---------------------
        # 3. Final conv
        self.fconv = nn.Conv2d(in_channels=self.hidden_size, out_channels=item_num,
                               kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, inputs, inputs_lengths):
        # ---------------------
        # 1. embed the input
        context_embedding = self.item_embeddings(inputs)

        dilate_output = context_embedding
        for dilated in self.res_dil_convs:
            dilate_output = dilated(dilate_output)

        dilate_output = dilate_output.permute(0, 2, 1)
        dilate_output = dilate_output.unsqueeze(dim=2)
        conv_output = self.fconv(dilate_output)
        conv_output = conv_output.squeeze(dim=2)
        conv_output = conv_output.permute(0, 2, 1)
        return conv_output
    
    def create_residual(self, blocktype, dilation, in_channels, out_channels, kernel_size, block_id, prev_blk):
        if blocktype == res_blk.ResidualBlockType.SIMPLE.value:
            return res_blk.SimpleResidualBlock(dilation, in_channels, out_channels, kernel_size, self.model_params['seqlen'], block_id, prev_blk)
        elif blocktype == res_blk.ResidualBlockType.CROSS_LAYER.value:
            return res_blk.CpResidualBlockCrossLayer(dilation, in_channels, out_channels, kernel_size, self.model_params['seqlen'], block_id, prev_blk)
        elif blocktype == res_blk.ResidualBlockType.CROSS_BLOCK.value:
            return res_blk.SimpleResidualBlock(dilation, in_channels, out_channels, kernel_size, self.model_params['seqlen'], block_id, prev_blk)
        elif blocktype == res_blk.ResidualBlockType.ADJACENT_LAYER.value:
            return res_blk.CpResidualBlockAdjacentLayer(dilation, in_channels, out_channels, kernel_size, self.model_params['seqlen'], block_id, prev_blk)
        elif blocktype == res_blk.ResidualBlockType.ADJACENT_BLOCK.value:
            return res_blk.CpResidualBlockAdjacentBlock(dilation, in_channels, out_channels, kernel_size, self.model_params['seqlen'], block_id, prev_blk)


class CpNltNetFullEvaluator(train_eval.Evaluator):
    def __init__(self, device, args, data_directory, state_size, item_num):
        super(CpNltNetFullEvaluator, self).__init__(device, args, data_directory, state_size, item_num)
        self.softmax = torch.nn.Softmax()

    def get_prediction(self, model, states, len_states, device):
        prediction = model(states.to(device).long(), len_states.to(device).long())
        return prediction

    def create_val_loader(self, filepath):
        eval_sessions = pd.read_pickle(filepath)
        sessions = eval_sessions.values
        inputs = sessions[:, 0:-1]
        inputs = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in inputs]).long()
        len_inputs = [np.count_nonzero(i) for i in inputs]
        len_inputs = torch.from_numpy(np.fromiter(len_inputs, dtype=np.long)).long()
        targets = sessions[:, 1:]
        targets = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in targets]).long()
        val_data = TensorDataset(inputs, len_inputs, targets)
        val_loader = DataLoader(val_data, shuffle=True, batch_size=self.args.batch_size)
        return val_loader

    def evaluate(self, model, val_or_test):
        topk = [5, 10, 15, 20]
        val_loader = self.create_val_loader(os.path.join(self.data_directory,
                                                    val_or_test + '_sessions.df'))
        hit = [0, 0, 0, 0]
        lens = [0, 0, 0, 0]
        ndcg = [0, 0, 0, 0]

        with torch.no_grad():
            model.eval()
            logger.log('Evaluation started...', self.args.modelname)
            for batch_idx, (states, len_states, target) in enumerate(val_loader):
                target = target[:, -1]
                prediction = self.get_prediction(model, states, len_states, self.device)
                del states
                del len_states
                torch.cuda.empty_cache()
                sorted_list = np.argsort(prediction[:, -1, :].tolist())

                def cal_hit(sorted_list, topk, true_items, hit, ndcg):
                    for i in range(len(topk)):
                      rec_list = sorted_list[:, -topk[i]:]
                      for j in range(len(true_items)):
                        if true_items[j] in rec_list[j]:
                            rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                            hit[i] += 1.0
                            ndcg[i] += 1.0 / np.log2(rank + 1)
                        lens[i] += 1
                cal_hit(sorted_list, topk, target.tolist(), hit, ndcg)

                if batch_idx % 200 == 0:
                  logger.log('Evaluated {} / {}'.format(batch_idx, len(val_loader)), self.args.modelname)
            logger.log('#############################################################', self.args.modelname)

            val_acc = 0
            for i in range(len(topk)):
                hr = hit[i] / float(lens[i])
                ng = ndcg[i] / float(lens[i])
                val_acc = val_acc + hr + ng
                logger.log('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', self.args.modelname)

                logger.log('hr ndcg @ %d : %f, %f' % (topk[i], hr, ng), self.args.modelname)

            logger.log('#############################################################', self.args.modelname)
            if val_or_test == "val":
                return val_acc


class CpNltNetFullTrainer(train_eval.Trainer):

    def create_model(self, model_params):
        cpNltNetTorch = CpNltNetFull(hidden_size=self.args.hidden_factor, item_num=self.item_num,
                                 device=self.device, model_params=model_params)
        total_params = sum(p.numel() for p in cpNltNetTorch.parameters() if p.requires_grad)
        print(cpNltNetTorch)
        return cpNltNetTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        out2d = torch.reshape(out, [-1, self.item_num])
        return out2d

    def preprocess_target(self, target):
        target = torch.reshape(target, [-1])
        return target

    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        cpNlt_evaluator = CpNltNetFullEvaluator(device, args, data_directory, state_size, item_num)
        return cpNlt_evaluator

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


TRAIN = 'train'
TEST = 'test'


def train_model(args, device, state_size, item_num, model_name, model_param, train_loader):
    nlt_trainer = CpNltNetFullTrainer(model_name, args, device, state_size, item_num, model_param)
    nlt_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num, model_name, model_params):
    cpNltTorch = CpNltNetFull(hidden_size=args.hidden_factor, item_num=item_num, device=device, model_params=model_params)
    checkpoint_handler = train_eval.CheckpointHandler(model_name, device)
    optimizer = torch.optim.Adam(cpNltTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, cpNltTorch, optimizer)
    cpNltTorch.to(device)

    cpNlt_evaluator = CpNltNetFullEvaluator(device, args, data_directory, state_size, item_num)
    cpNlt_evaluator.evaluate(cpNltTorch, 'test')


#prepare a dataloader as described in nextlt paper: input :{1,2,3,4,5} output:{2,3,4,5,6}
def prepare_dataloader_whole(data_dir, batch_size):
    basepath = os.path.dirname(__file__)
    sessions_df = pd.read_pickle(os.path.abspath(os.path.join(basepath, data_dir, 'train_sessions.df')))
    sessions = sessions_df.values
    inputs = sessions[:, 0:-1]
    inputs = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in inputs]).long()
    len_inputs = [np.count_nonzero(i) for i in inputs]
    len_inputs = torch.from_numpy(np.fromiter(len_inputs, dtype=np.long)).long()
    targets = sessions[:, 1:]
    targets = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in targets]).long()
    train_data = TensorDataset(inputs, len_inputs, targets)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    return train_loader


if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    device = set_device()
    data_directory = args.data
    model_name = args.modelname

    state_size, item_num = train_eval.get_stats(data_directory)
    train_loader = prepare_dataloader_whole(data_directory, args.batch_size)

    model_param = {
        'dilated_channels': args.hidden_factor,  # larger is better until 512 or 1024
        'dilations': [int(item) for item in args.dilations.split(',')],  # YOU should tune this hyper-parameter, refer to the paper.
        'kernel_size': 3,
        'resblocktype': args.resblocktype,
        'seqlen' : state_size
        # 'param-share' : ''
    }

    
    
    if args.mode.lower() == TRAIN:
        train_model(args, device, state_size, item_num, model_name, model_param, train_loader)
    else:
        test_model(device, args, data_directory, state_size, item_num, model_name, model_param)


