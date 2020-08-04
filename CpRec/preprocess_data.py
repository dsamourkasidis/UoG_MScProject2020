import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from utility import to_pickled_df

def preprocess_movielens20(path):
    data_file = path
    sessions = list(open(data_file, "r").readlines())
    sessions = [s for s in sessions]
    items_voc = []

    padid = 0
    items_voc.append(padid)
    max_seq_length = max([len(x.split(",")) for x in sessions])

    for sample in sessions:
        for item in sample.strip().split(","):
                items_voc.append(int(item))
    items_voc = set(items_voc)

    item2id = dict(zip(items_voc, range(len(items_voc))))

    #INFO_LOG("Vocab size:{}".format(self.size()))
    def sample2id(session):
        sample2id = []
        for i in session.strip().split(','):
            sample2id.append(item2id[int(i)])
        sample2id = ([padid] * (max_seq_length - len(sample2id))) + sample2id

        return sample2id

    def getSamplesid(sessions):
        samples2id = []
        for session in sessions:
            samples2id.append(sample2id(session))

        return samples2id

    items = np.array(getSamplesid(sessions))
    return items, max_seq_length, len(items_voc)


def split_sessions(sessions):
    train_val_sessions, test_sessions = train_test_split(sessions, train_size=.9)
    train_sessions, val_sessions = train_test_split(sessions, train_size=.9)
    return train_sessions, val_sessions, test_sessions

def split_sessions_val(sessions):
    train_sessions, val_sessions = train_test_split(sessions, train_size=.8)
    return train_sessions, val_sessions

if __name__ == '__main__':
    data_dir = '../data/ML100'
    datacsv = 'mllatest_ls100.csv'
    sessions, max_len, item_num = preprocess_movielens20(data_dir + '/' + datacsv)

    dic = {'state_size': [max_len], 'item_num': [item_num]}
    data_statis = pd.DataFrame(data=dic)
    to_pickled_df(data_dir, data_statis=data_statis)

    train_sessions, val_sessions = split_sessions_val(sessions)
    train_sessions_df = pd.DataFrame(train_sessions)
    to_pickled_df(data_dir, train_sessions=train_sessions_df)
    val_sessions_df = pd.DataFrame(val_sessions)
    to_pickled_df(data_dir, val_sessions=val_sessions_df)


    # train_sessions, val_sessions, test_sessions = split_sessions(sessions)
    # train_sessions_df = pd.DataFrame(train_sessions)
    # to_pickled_df(data_dir, train_sessions=train_sessions_df)
    # val_sessions_df = pd.DataFrame(val_sessions)
    # to_pickled_df(data_dir, val_sessions=val_sessions_df)
    # test_sessions_df = pd.DataFrame(test_sessions)
    # to_pickled_df(data_dir, test_sessions=test_sessions_df)

