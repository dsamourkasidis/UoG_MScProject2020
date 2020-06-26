import torch
import torch.nn as nn
import argparse
from utility import extract_axis_1_torch, normalize, set_device
from SASRecModulesTorch import multihead_attention,feedforward
import train_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Run SASRec.")

    parser.add_argument('--mode', default='train',
                        help='Train or test the model. "train" or "test"')
    parser.add_argument('--epochs', type=int, default=4,
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
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    return parser.parse_args()


class SASRecTorch(nn.Module):
    def __init__(self, hidden_size,item_num,state_size,device,num_blocks,num_heads,dropout_rate):
        super(SASRecTorch, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.device = torch.device("cpu")
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate=dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num+1,
            embedding_dim=self.hidden_size,
        )
        self.pos_embeddings = nn.Embedding(
            num_embeddings=self.state_size,
            embedding_dim=self.hidden_size,
        )
        
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        nn.init.normal_(self.pos_embeddings.weight, 0, 0.01)
        
        #Multihead Attention Layer
        #x`self.multihead_attention = multihead_attention()
        
        #Feedforward Layer
        self.feedforward = feedforward(num_units=[self.hidden_size,self.hidden_size],
                                     dropout_rate=self.dropout_rate)
        
        #dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        #Fully connected Layer
        self.fc1 = nn.Linear(self.hidden_size,self.item_num)
        
    def forward(self, inputs, inputs_lengths):
        input_emb = self.item_embeddings(inputs)
        pos_emb_input = torch.cat(inputs.size(0)*[torch.range(start=0,end=inputs.size(1)-1).unsqueeze(0)])
        pos_emb_input = pos_emb_input.long()
        pos_emb = self.pos_embeddings(pos_emb_input)
        x = input_emb+pos_emb
            
        x = self.dropout(x)
            
        mask = torch.ne(inputs, self.item_num).float().unsqueeze(-1)
        x *= mask
            
        for i in range (self.num_blocks):
            x = multihead_attention(queries=normalize(x),keys=x,
                                             num_units=self.hidden_size,
                                             num_heads=self.num_heads,
                                             dropout_rate=self.dropout_rate,
                                             causality=True)
        x = self.feedforward(normalize(x))
        x *= mask
            
        x = normalize(x)
        out = extract_axis_1_torch(x,inputs_lengths-1)
        out = self.fc1(out)
        return out
            
                
        
class SASRecEvaluator(train_eval.Evaluator):
    def get_prediction(self, model, states, len_states, device):
        prediction = model(states.to(device).long(), len_states.to(device).long())
        return prediction


class SASRecTrainer(train_eval.Trainer):

    def create_model(self):
        sasrecTorch = SASRecTorch(hidden_size=self.args.hidden_factor, item_num=self.item_num,
                            state_size=self.state_size, device=self.device,num_blocks=self.args.num_blocks,
                            num_heads=self.args.num_heads,dropout_rate=self.args.dropout_rate)
        return sasrecTorch

    def get_model_out(self, state, len_state):
        out = self.model(state, len_state)
        return out

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


def train_model(args, device, state_size, item_num):
    sasrec_trainer = SASRecTrainer('caser_RC15', args, device, state_size, item_num)
    sasrec_trainer.train(train_loader)


def test_model(device, args, data_directory, state_size, item_num):
    sasrecTorch = SASRecTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device,num_blocks=args.num_blocks,
                            num_heads=args.num_heads,dropout_rate=args.dropout_rate)
    checkpoint_handler = train_eval.CheckpointHandler('sasrec_RC15', device)
    optimizer = torch.optim.Adam(sasrecTorch.parameters(), lr=args.lr)
    _, _ = checkpoint_handler.load_from_checkpoint(True, sasrecTorch, optimizer)
    sasrecTorch.to(device)

    sasrec_evaluator = SASRecEvaluator(device, args, data_directory, state_size, item_num)
    sasrec_evaluator.evaluate(sasrecTorch, 'test')


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
