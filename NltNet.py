import torch
import torch.nn as nn
import argparse
from utility import set_device
import train_eval
import NltNetModules as nlt_modules
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised NltNet.")

    parser.add_argument('--mode', default='train',
                        help='Train or test the model. "train" or "test"')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data/Steam',
                        help='data directory')
    parser.add_argument('--resume', type=int, default=1,
                        help='flag for resume. 1: resume training; 0: train from start')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--dilations', type=str, default="1,4,1,4",
                        help='dilations for res blocks. eg "1,4,1,4"')
    parser.add_argument('--modelname', type=str, default="nltnet-ml20-16-64-256",
                        help='model name. eg for checkpoint filename')
    return parser.parse_args()


class NltNet(nn.Module):
    def __init__(self, hidden_size, item_num, device, model_params):
        super(NltNet, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.model_params = model_params

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num+1,
            embedding_dim=self.hidden_size,
            # padding_idx=padding_idx
        )
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        
        residual_blocks = []
        for layer_id, dil in enumerate(self.model_params['dilations']):
            bl = nlt_modules.SimpleResidualBlock(dil, self.hidden_size, self.model_params['dilated_channels'],
                                                 self.model_params['kernel_size'], layer_id)
            residual_blocks.append(bl)
        self.res_dil_convs = nn.ModuleList(residual_blocks)

        self.fconv = nn.Conv2d(in_channels=self.hidden_size, out_channels=item_num,
                               kernel_size=1, padding=0, dilation=1, bias=True)

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

        dilate_output = dilate_output.permute(0, 2, 1)
        dilate_output = dilate_output.unsqueeze(dim=2)
        conv_output = self.fconv(dilate_output)
        conv_output = conv_output.squeeze(dim=2)
        conv_output = conv_output.permute(0, 2, 1)
        return conv_output


class NltNetEvaluator(train_eval.Evaluator):
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


class NltNetTrainer(train_eval.Trainer):

    def create_model(self, model_params):
        nltNet = NltNet(hidden_size=self.args.hidden_factor, item_num=self.item_num, device=self.device,
                        model_params=model_params)
        print(nltNet)
        total_params = sum(p.numel() for p in nltNet.parameters() if p.requires_grad)
        print(total_params)
        return nltNet

    def preprocess_target(self, target):
        target = torch.reshape(target, [-1])
        return target

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        out2d = torch.reshape(out, [-1, self.item_num])
        return out2d

    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        nlt_evaluator = NltNetEvaluator(device, args, data_directory, state_size, item_num)
        return nlt_evaluator

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


TRAIN = 'train'
TEST = 'test'


def train_model(args, device, state_size, item_num, model_name, model_param, train_loader):
    nlt_trainer = NltNetTrainer(model_name, args, device, state_size, item_num, model_param)
    nlt_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num, model_name, model_params):
    nltNet = NltNet(hidden_size=args.hidden_factor, item_num=item_num, device=device,
                    model_params=model_params)
    checkpoint_handler = train_eval.CheckpointHandler(model_name, device)
    optimizer = torch.optim.Adam(nltNet.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, nltNet, optimizer)
    nltNet.to(device)

    nltnet_evaluator = NltNetEvaluator(device, args, data_directory, state_size, item_num)
    nltnet_evaluator.evaluate(nltNet, 'val')


# prepare a dataloader as described in nextlt paper: input :{1,2,3,4,5} output:{2,3,4,5,6}
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
        'dilations': [int(item) for item in args.dilations.split(',')],
        # YOU should tune this hyper-parameter, refer to the paper.
        'kernel_size': 3,
        'seqlen': state_size
        # 'param-share' : ''
    }

    if args.mode.lower() == TRAIN:
        train_model(args, device, state_size, item_num, model_name, model_param, train_loader)
    else:
        test_model(device, args, data_directory, state_size, item_num, model_name, model_param)


