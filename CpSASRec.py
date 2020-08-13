import torch
import torch.nn as nn
import argparse
from utility import extract_axis_1_torch, normalize, set_device
from SASRecModules import LayerNorm, SelfAttentionBlock, SelfAttentionBlockAdjacentBlock
import train_eval
import os
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run SASRec.")

    parser.add_argument('--mode', default='train',
                        help='Train or test the model. "train" or "test"')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data\ML100',
                        help='data directory')
    parser.add_argument('--resume', type=int, default=1,
                        help='flag for resume. 1: resume training; 0: train from start')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--modelname', type=str, default="SASrec_256_64",
                        help='model name. eg for checkpoint filename')
    return parser.parse_args()


class SASRecTorch(nn.Module):
    def __init__(self, item_num, state_size, device, model_params):
        super(SASRecTorch, self).__init__()
        self.hidden_size = model_params['hidden_factor']
        self.item_num = int(item_num)
        self.state_size = state_size
        self.device = device
        self.num_blocks = model_params['num_blocks']
        self.num_heads = model_params['num_heads']
        self.dropout_rate = model_params['dropout_rate']
        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=self.hidden_size,
        )
        self.pos_embeddings = nn.Embedding(
            num_embeddings=self.state_size,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        nn.init.normal_(self.pos_embeddings.weight, 0, 0.01)

        # Self Attention Blocks
        self_att_blks_list = []
        bl = SelfAttentionBlockAdjacentBlock(self.hidden_size, self.num_heads, self.dropout_rate, state_size, 0, None)
        self_att_blks_list.append(bl)
        for i in range(self.num_blocks-1):
            bl = SelfAttentionBlockAdjacentBlock(self.hidden_size, self.num_heads, self.dropout_rate,
                                                 state_size, i+1, bl)
            self_att_blks_list.append(bl)

        self.self_att_blks = nn.ModuleList(self_att_blks_list)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        self.f_layer_norm = LayerNorm(self.hidden_size)
        # Fully connected Layer
        self.fc1 = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, inputs, inputs_lengths):
        input_emb = self.item_embeddings(inputs)
        pos_emb_input = inputs.size(0) * [torch.arange(start=0, end=inputs.size(1)).unsqueeze(0)]
        pos_emb_input = torch.cat(pos_emb_input)
        pos_emb_input = pos_emb_input.long().to(self.device)
        pos_emb = self.pos_embeddings(pos_emb_input)
        x = input_emb + pos_emb

        x = self.dropout(x)

        padid = 0
        mask = torch.ne(inputs, padid).float().unsqueeze(-1)
        x = x * mask

        for i in range(self.num_blocks):
            x = self.self_att_blks[i](x)
            x = x * mask

        x = self.f_layer_norm(x)
        # out = self.extract_unpadded(x, inputs_lengths-1)
        # out = self.fc1(out)
        indcs = torch.ones(inputs.size(0)) * self.state_size-1
        out = self.extract_unpadded(x, indcs.long().to(self.device) - 1)
        out = self.fc1(out)
        return out

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


class SASRecEvaluator(train_eval.Evaluator):
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
                sorted_list = np.argsort(prediction.tolist())

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


class SASRecTrainer(train_eval.Trainer):

    def create_model(self, model_params):
        sasrecTorch = SASRecTorch(item_num=item_num, state_size=state_size, device=device, model_params=model_params)
        return sasrecTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        out2d = torch.reshape(out, [-1, self.item_num])
        return out2d

    def preprocess_target(self, target):
        target = torch.reshape(target, [-1])
        return target

    def get_evaluator(self, device, args, data_directory, state_size, item_num):
        sasrec_evaluator = SASRecEvaluator(device, args, data_directory, state_size, item_num)
        return sasrec_evaluator

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion


TRAIN = 'train'
TEST = 'test'


def train_model(args, device, state_size, item_num, model_name, model_param, train_loader):
    sasrec_trainer = SASRecTrainer(model_name, args, device, state_size, item_num, model_param)
    sasrec_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num, model_name, model_params):
    sasrecTorch = SASRecTorch(item_num=item_num, state_size=state_size, device=device, model_params=model_params)

    checkpoint_handler = train_eval.CheckpointHandler(model_name, device)
    optimizer = torch.optim.Adam(sasrecTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, sasrecTorch, optimizer)
    sasrecTorch.to(device)

    sasrec_evaluator = SASRecEvaluator(device, args, data_directory, state_size, item_num)
    sasrec_evaluator.evaluate(sasrecTorch, 'val')


# prepare a dataloader: input :{1,2,3,4,5} output:{6}
def prepare_dataloader_whole(data_dir, batch_size):
    basepath = os.path.dirname(__file__)
    sessions_df = pd.read_pickle(os.path.abspath(os.path.join(basepath, data_dir, 'train_sessions.df')))
    sessions = sessions_df.values
    inputs = sessions[:, 0:-1]
    inputs = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in inputs]).long()
    len_inputs = [np.count_nonzero(i) for i in inputs]
    len_inputs = torch.from_numpy(np.fromiter(len_inputs, dtype=np.long)).long()
    targets = sessions[:, -1]
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
        'hidden_factor': args.hidden_factor,  # larger is better until 512 or 1024
        'num_blocks': args.num_blocks,
        'num_heads': args.num_heads,
        'dropout_rate': args.dropout_rate
    }

    if args.mode.lower() == TRAIN:
        train_model(args, device, state_size, item_num, model_name, model_param, train_loader)
    else:
        test_model(device, args, data_directory, state_size, item_num, model_name, model_param)
