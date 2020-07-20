import torch
import torch.nn as nn
import argparse
from utility import set_device
import train_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised NltNet.")

    parser.add_argument('--mode', default='train',
                        help='Train or test the model. "train" or "test"')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data/RetailRocket',
                        help='data directory')
    parser.add_argument('--resume', type=int, default=1,
                        help='flag for resume. 1: resume training; 0: train from start')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    return parser.parse_args()


class NltNetFullTorch(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, device, gru_layers=1):
        super(NltNetTorch, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.conv_param = {
            'dilated_channels': 64,  # larger is better until 512 or 1024
            'dilations': [1, 2, 1, 2, 1, 2, ],  # YOU should tune this hyper-parameter, refer to the paper.
            'kernel_size': 3,
        }

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num+1,
            embedding_dim=self.hidden_size,
            # padding_idx=padding_idx
        )
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        
        residual_blocks = []
        for layer_id, dil in enumerate(self.model_params['dilations']):
            residual_blocks.append(res_blk.SimpleResidualBlock(dil, self.hidden_size, self.model_params['dilated_channels'],
                                                   self.model_params['kernel_size'], layer_id+1))
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

        conv_output = self.fconv(dilate_output)
        conv_output = conv_output.squeeze(dim=2)
        conv_output = conv_output.permute(0, 2, 1)
        return conv_output


class ResidualBlock(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, causal=True):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               padding=0, dilation=dilation, bias=True)
        self.layer_norm1 = None
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               padding=0, dilation=2*dilation, bias=True)
        self.layer_norm2 = None
        self.rel2 = nn.ReLU()

    def forward(self, inputs):
        padding = [0, 0, (self.kernel_size - 1) * self.dilation, 0, 0, 0]
        padded = torch.nn.functional.pad(inputs, padding)
        # pytorch expect BCHW input that is why we permute and add height in dim 2
        padded = padded.permute(0, 2, 1)
        input_expanded = padded.unsqueeze(dim=2)
        y = self.conv1(input_expanded)
        y = y.squeeze(dim=2)
        if self.layer_norm1 is None:
            self.layer_norm1 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm1(y)
        y = self.rel1(y)

        padding = [(self.kernel_size - 1) * 2*self.dilation, 0, 0, 0, 0, 0]
        padded = torch.nn.functional.pad(y, padding)
        padded = padded.unsqueeze(dim=2)
        y = self.conv2(padded)
        y = y.squeeze(dim=2)
        if self.layer_norm2 is None:
            self.layer_norm2 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm2(y)
        y = self.rel2(y)


        y = y.permute(0, 2, 1)
        return inputs + y


class NltNetEvaluator(train_eval.Evaluator):
    def get_prediction(self, model, states, len_states, device):
        prediction = model(states.to(device).long(), len_states.to(device).long())
        return prediction


class NltNetTrainer(train_eval.Trainer):

    def create_model(self, model_params):
        nltNetTorch = NltNetTorch(hidden_size=self.args.hidden_factor, item_num=self.item_num,
                            state_size=self.state_size, device=self.device)
        print(nltNetTorch)
        total_params = sum(p.numel() for p in nltNetTorch.parameters() if p.requires_grad)
        print(total_params)
        return nltNetTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        return out

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


def train_model(args, device, state_size, item_num):
    nlt_trainer = NltNetTrainer('nltNet_RetailRocket', args, device, state_size, item_num, None)
    nlt_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num):
    nltTorch = NltNetTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device)
    checkpoint_handler = train_eval.CheckpointHandler('nltNet_RC15', device)
    optimizer = torch.optim.Adam(nltTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, nltTorch, optimizer)
    nltTorch.to(device)

    gru_evaluator = NltNetEvaluator(device, args, data_directory, state_size, item_num)
    gru_evaluator.evaluate(nltTorch, 'test')


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


