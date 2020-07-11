import torch
import torch.nn as nn
import argparse
import pandas as pd
import os
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from utility import pad_history, calculate_hit
import pickle

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
    parser.add_argument('--discount', type=float, default=0.5,
                        help='Discount factor for RL.')
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
        self.fc1 = nn.Linear(final_dim,item_num)
        self.fc2 = nn.Linear(final_dim,item_num)
        
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
        q_learning_output = self.fc1(out)
        supervised_output = self.fc2(out)
        
        #return q_learning_output
        
        return q_learning_output,supervised_output
    
class Double_qlearning(nn.Module):
    def __init__(self,item_num):
        super(Double_qlearning, self).__init__()
        self.item_num = item_num
        self.fc = nn.Linear(self.item_num,1)
        
    def forward(self,hidden_states,actions,rewards,discount,targetQs_s,is_done):
        total_loss = 0
        batch_size = len(hidden_states)
        discount = discount.item()
        for i in range(batch_size):
            done = is_done[i]
            if (done):
                continue
            else:
                state = hidden_states[i]
                #action = actions[i]
                reward = rewards[i]
                #targetQs_ = targetQs_s[i]
                next_state = targetQs_s[i]
                q_value = nn.functional.relu(self.fc(state))
                next_state_q_value = nn.functional.relu(self.fc(next_state))
                target = reward + (discount*next_state_q_value)
                q_loss = abs(target - q_value)
                total_loss += q_loss
        return (total_loss/batch_size)
        

def set_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def prepare_dataloader(data_directory):
    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    replay_buffer_dic = replay_buffer.to_dict()
    
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.long)).long()
    
    next_states = replay_buffer_dic['next_state'].values()
    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.long)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.long)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool))
    
    train_data = TensorDataset(states, len_states, actions,next_states,
                               len_next_states,is_buy,is_done)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)
    return train_loader


def get_stats(data_dir):
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    return state_size, item_num


def evaluate(model, data_directory, args):
    topk = [5, 10, 15, 20]
    reward_click = args.r_click
    reward_buy = args.r_buy
    eval_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    model.eval()
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
            evaluated += 1
        states = torch.stack([torch.from_numpy(np.array(i, dtype=np.long)) for i in states]).long()
        len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.long)).long()
        _,prediction = model(states.to(device).long(), len_states.to(device).long())
        del states
        del len_states
        torch.cuda.empty_cache()
        sorted_list = np.argsort(prediction.tolist())
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward,
                      hit_clicks, ndcg_clicks, hit_purchase, ndcg_purchase)
    print('#############################################################')
    print('total clicks: %d, total purchases:%d' % (total_clicks, total_purchase))
    val_acc = 0
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        val_acc = val_acc + hr_click + hr_purchase + ng_click + ng_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    print('#############################################################')
    return val_acc

if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    device = set_device()
    data_directory = args.data

    state_size, item_num = get_stats(data_directory)
    train_loader = prepare_dataloader(data_directory)

    caserTorch1 = CaserTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device,num_filters=args.num_filters,
                            filter_sizes=args.filter_sizes,dropout_rate=args.dropout_rate)
    
    caserTorch2 = CaserTorch(hidden_size=args.hidden_factor, item_num=item_num,
                        state_size=state_size, device=device,num_filters=args.num_filters,
                            filter_sizes=args.filter_sizes,dropout_rate=args.dropout_rate)

    double_qlearning = Double_qlearning(item_num=item_num)
    criterion = nn.CrossEntropyLoss()
    params1 = list(caserTorch1.parameters())+list(double_qlearning.parameters())
    optimizer1 = torch.optim.Adam(params1, lr=args.lr)
    
    params2 = list(caserTorch2.parameters())+list(double_qlearning.parameters())
    optimizer2 = torch.optim.Adam(params2, lr=args.lr) 

    #checkpoint_file = 'gru__best.pth.tar'
    #start_epoch, min_loss = load_from_checkpoint(args, checkpoint_file, gruTorch, optimizer, device)
    caserTorch1.to(device)
    caserTorch2.to(device)
    double_qlearning.to(device)
    
    reward_click = args.r_click
    reward_buy = args.r_buy

    # Start training loop
    epoch_times = []
    total_step = 0
    best_val_acc = 0
    for epoch in range(0, args.epochs):
        caserTorch1.train()
        caserTorch2.train()
        start_time = time.time()
        avg_loss = 0.
        counter = 0
        for state,len_state,action,next_state,len_next_state,is_buy,is_done in train_loader:
            counter += 1
            pointer = np.random.randint(0, 2)
            if pointer == 0:
                main_QN = caserTorch1
                target_QN = caserTorch2
                optimizer = optimizer1
                model_name = "Caser 1"
            else:
                main_QN = caserTorch2
                target_QN = caserTorch1
                optimizer = optimizer2
                model_name = "Caser 2"
                
            main_QN.zero_grad()
            target_QN.zero_grad()
            
            with torch.no_grad():
                target_Qs,_ = target_QN(next_state.to(device).long(), len_next_state.to(device).long())
                #target_Qs_selector,_ = main_QN(next_state.to(device).long(), len_next_state.to(device).long())
               
            reward = []
            for k in range(len(is_buy)):
                reward.append(reward_buy if is_buy[k] == 1 else reward_click)
            reward = torch.tensor(reward)
            discount = args.discount
            discount = torch.tensor(discount)
            
            main_QN.zero_grad()
            q_out,supervised_out = main_QN(state.to(device).long(), len_state.to(device).long())
            supervised_loss = criterion(supervised_out, action.to(device).long())

            double_qlearning.zero_grad()
            q_loss = double_qlearning(q_out.to(device),action.to(device),reward.to(device),discount.to(device),
                                      target_Qs.to(device),is_done.to(device))

            sqn_loss = supervised_loss + q_loss
            optimizer.zero_grad()
            sqn_loss.backward()
            optimizer.step()

            total_step += 1
            if total_step % 200 == 0:
                print ("Model is ",model_name)
                print("Supervised loss is %.3f, q-loss is %.3f" % (supervised_loss.item(), q_loss))
                print("Epoch {}......Batch: {}/{}....... Loss: {}".format(epoch, counter,
                                                                          len(train_loader),
                                                                          sqn_loss.item()))
            if total_step % 2000 == 0:
                print ("Evaluating Main Model")
                val_acc_main = evaluate(main_QN, data_directory, args)
                main_QN.train()
                print ("Evaluating Target Model")
                val_acc_target = evaluate(target_QN, data_directory, args)
                target_QN.train()
                print ("Best accuracy so far: ",best_val_acc)
                if (val_acc_main>best_val_acc or val_acc_target>best_val_acc):
                    if (val_acc_main>=val_acc_target):
                        best_val_acc = val_acc_main
                        print ("Main model is the best, so far!")
                        print ("New best accuracy is: %.3f"%(best_val_acc))
                        pickle.dump(main_QN, open('Casertorch_model_SQN2.pth', 'wb'))
                    else:
                        best_val_acc = val_acc_target
                        print ("Target model is the best, so far!")
                        print ("New best accuracy is: %.3f"%(best_val_acc))
                        pickle.dump(target_QN, open('Casertorch_model_SQN2.pth', 'wb'))
        current_time = time.time()
        print("Epoch {}/{} Done".format(epoch, args.epochs))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    pickle.dump(caserTorch1, open('Casertorch_model.pth', 'wb'))