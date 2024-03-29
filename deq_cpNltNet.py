import sys, os, inspect

sys.path.append('../../')
import torch
import torch.nn as nn
import argparse
import deq_modules.deq_residual_block as deq_res_blk
import train_eval
from utility import set_device, calculate_simple_hit
import deq_modules.deq_residual_block as deq_res_blk
from deq_modules.deq_cpNltNet_module import CpNltNetDEQModule
import pandas as pd
import numpy as np
import os
import logger
from torch.utils.data import TensorDataset, DataLoader
import copy
import time
from train_eval import CheckpointHandler


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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=8,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dilations', type=str, default="1,4",
                        help='dilations for res blocks. eg "1,4"')
    parser.add_argument('--modelname', type=str, default="deqCpnltnet_ML100",
                        help='model name. eg for checkpoint filename')
    return parser.parse_args()


class DEQCpNltNet(nn.Module):
    def __init__(self, hidden_size, item_num, device, model_params):
        super(DEQCpNltNet, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.model_params = model_params
        self.kernel_size = self.model_params['kernel_size']

        # --------------------
        # 1. Embeddings
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        self.inject_conv = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.model_params['dilated_channels'],
                                     kernel_size=(1, self.kernel_size), padding=0, dilation=1, bias=True)
        # ---------------------
        # 2. DEQ Residual block
        self.func = deq_res_blk.DEQCpResidualBlockAdjacentBlock(self.model_params['dilations'],
                                                                self.hidden_size, self.model_params['dilated_channels'],
                                                                self.kernel_size, 0)
        self.func_copy = copy.deepcopy(self.func)
        for params in self.func_copy.parameters():
            params.requires_grad_(False)  # Turn off autograd for func_copy

        self.deq = CpNltNetDEQModule(self.func, self.func_copy)

        # ---------------------
        # 3. Final conv
        self.fconv = nn.Conv2d(in_channels=self.hidden_size, out_channels=item_num,
                               kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, inputs, inputs_lengths, train_step):
        # copy the weights from func to func_copy
        self.func_copy.copy(self.func)

        # embed the input
        context_embedding = self.item_embeddings(inputs)

        # create injected input
        z1s = context_embedding.permute(0, 2, 1)
        padding = [(self.kernel_size - 1), 0, 0, 0, 0, 0]
        padded = torch.nn.functional.pad(z1s, padding)
        us = padded.unsqueeze(dim=2)
        us = self.inject_conv(us)
        us = us.squeeze(dim=2)

        # deq the residual block
        dilate_output = self.deq(z1s, us, train_step=train_step)

        # final conv for output
        dilate_output = dilate_output.unsqueeze(dim=2)
        conv_output = self.fconv(dilate_output)
        conv_output = conv_output.squeeze(dim=2)
        conv_output = conv_output.permute(0, 2, 1)
        return conv_output


class DEQcpNltNetEvaluator(train_eval.Evaluator):
    def __init__(self, device, args, data_directory, state_size, item_num):
        super(DEQcpNltNetEvaluator, self).__init__(device, args, data_directory, state_size, item_num)
        self.softmax = torch.nn.Softmax()

    def get_prediction(self, model, states, len_states, device):
        pass

    def get_deq_prediction(self, model, states, len_states, device, train_step):
        prediction = model(states.to(device).long(), len_states.to(device).long(), train_step)
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
        pass

    def deq_evaluate(self, model, val_or_test, train_step):
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
                prediction = self.get_deq_prediction(model, states, len_states, self.device, train_step)
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


class DEQcpNltNetTrainer(train_eval.Trainer):

    def create_model(self, model_params):
        deq_cpNltNetTorch = DEQCpNltNet(hidden_size=self.args.hidden_factor, item_num=self.item_num,
                                     device=self.device, model_params=model_params)
        total_params = sum(p.numel() for p in deq_cpNltNetTorch.parameters() if p.requires_grad)
        print(deq_cpNltNetTorch)
        return deq_cpNltNetTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        out2d = torch.reshape(out, [-1, self.item_num])
        return out2d

    def get_deqmodel_out(self, state, len_state, train_step):
        out = self.model(state, len_state, train_step)
        out2d = torch.reshape(out, [-1, self.item_num])
        return out2d

    def preprocess_target(self, target):
        target = torch.reshape(target, [-1])
        return target

    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        cpNlt_evaluator = DEQcpNltNetEvaluator(device, args, data_directory, state_size, item_num)
        return cpNlt_evaluator

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def train(self, train_loader):

        checkpoint_handler = CheckpointHandler(self.model_name, self.device)
        start_epoch, max_acc = checkpoint_handler.load_from_checkpoint(self.args.resume, self.model, self.optimizer)
        self.model.to(self.device)

        evaluator = self.get_evaluator(self.device, self.args, self.args.data, self.state_size, self.item_num)
        criterion = self.get_criterion()

        # Start training loop
        epoch_times = []
        total_step = 0
        for epoch in range(start_epoch, self.args.epochs):
            self.model.train()
            start_time = time.perf_counter()
            avg_loss = 0.
            counter = 0
            for state, len_state, target in train_loader:
                counter += 1
                self.model.zero_grad()
                self.optimizer.zero_grad()

                out = self.get_deqmodel_out(state.to(self.device).long(), len_state.to(self.device).long(), total_step)
                target = self.preprocess_target(target)
                loss = criterion(out, target.to(self.device).long())
                all_params = torch.cat([x.view(-1) for x in self.model.parameters()])
                lambda2 = 0.001
                l2_regularization = lambda2 * torch.norm(all_params, 2)
                loss = loss + l2_regularization

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.07)
                self.optimizer.step()

                if total_step % 500 == 0:
                    logger.log("Epoch {}......Batch: {}/{}....... Loss: {}".format(epoch, counter,
                                                                              len(train_loader),
                                                                              loss.item()), self.model_name)
                total_step += 1

            current_time = time.perf_counter()
            logger.log("Epoch {}/{} Done, Total Loss: {}".format(epoch, self.args.epochs, avg_loss / len(train_loader)), self.model_name)
            logger.log("Total Time Elapsed: {} seconds".format(str(current_time - start_time)), self.model_name)
            val_acc = evaluator.deq_evaluate(self.model, 'val', total_step)
            self.model.train()
            is_best = val_acc > max_acc
            max_acc = max(max_acc, val_acc)
            checkpoint_handler.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'max_acc': max_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)
            epoch_times.append(current_time - start_time)
        logger.log("Total Training Time: {} seconds".format(str(sum(epoch_times))), self.model_name)


TRAIN = 'train'
TEST = 'test'


def train_model(args, device, state_size, item_num, model_name, model_param, train_loader):
    nlt_trainer = DEQcpNltNetTrainer(model_name, args, device, state_size, item_num, model_param)
    nlt_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num, model_name, model_params):
    cpNltTorch = DEQCpNltNet(hidden_size=args.hidden_factor, item_num=item_num, device=device,
                              model_params=model_params)
    checkpoint_handler = train_eval.CheckpointHandler(model_name, device)
    optimizer = torch.optim.Adam(cpNltTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, cpNltTorch, optimizer)
    cpNltTorch.to(device)

    cpNlt_evaluator = DEQcpNltNetEvaluator(device, args, data_directory, state_size, item_num)
    cpNlt_evaluator.evaluate(cpNltTorch, 'val')


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
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if args.mode.lower() == TRAIN:
        train_model(args, device, state_size, item_num, model_name, model_param, train_loader)
    else:
        test_model(device, args, data_directory, state_size, item_num, model_name, model_param)


