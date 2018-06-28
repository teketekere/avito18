import pandas as pd
import numpy as np
import gc
import time
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from myutils import timer

def create_tensor(x):
    if x is not None:
        x_tensor = torch.Tensor(x)
    else:
        x_tensor = None
    return x_tensor

def create_dataloader(x, y, categorical, numeric, batch_size=10, shuffle=True):
    # do adohook for categorical features to keep columns seq
    # categorical
    cat_df = pd.DataFrame()
    for cat in categorical:
        cat_df[cat] = x[cat].astype('int32')
    cat_tensor = create_tensor(cat_df.values).long()

    # numeric
    num_tensor = create_tensor(x[numeric].values)

    # words
    res = []
    for t in x['seq_title_description']:
        if len(t) > 100:
            temp = t[:100]
        elif len(t) < 100:
            for i in range(100 - len(t)):
                t.append(0)
            temp = t
        else:
            temp = t
        temp = np.array([int(val) if val != '' else int(0) for val in temp])
        res.append(temp)
    res = np.array(res)
    print(res.shape)
    word_tensor = torch.from_numpy(res)

    # y
    y_tensor = create_tensor(y.values)

    if y_tensor is not None:
        tensor_dataset = TensorDataset(cat_tensor, num_tensor, word_tensor,  y_tensor)
    else:
        tensor_dataset = TensorDataset(cat_tensor, num_tensor, word_tensor)

    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    del cat_df, cat_tensor, num_tensor, y_tensor, tensor_dataset, res, word_tensor; gc.collect()
    return dataloader

class GRUNet(nn.Module):
    def __init__(self, categorical_dict, batch_size, numeric, word_embed, rnn_input=300, hidden_state=64, hidden_layers=100):
        '''
        categorical_dict = {'cat1': (23, 16)}
        cat1: categorical variable column name
        23: (unique number of the cat1 elements) + 1
        16: Embedding dimension
        '''
        super(GRUNet, self).__init__()
        self.categorical_dict = categorical_dict
        self.batch_size = batch_size
        self.numeric = numeric
        self.rnn_input = rnn_input
        self.hidden_state = hidden_state
        self.hidden_layers = hidden_layers
        self.inputlayers = np.sum([val[1] for val in categorical_dict.values()]) + len(numeric) + 100*hidden_state

        if torch.cuda.is_available():
            self.embeddings = [nn.Embedding(val[0], val[1]).cuda() for val in categorical_dict.values()]
        else:
            self.embeddings = [nn.Embedding(val[0], val[1]) for val in categorical_dict.values()]
        words_weight = torch.Tensor(np.load(word_embed))
        self.embedding_words = nn.Embedding.from_pretrained(words_weight).cuda()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout1 = self.dropout1.cuda()
        self.gru = nn.GRU(rnn_input, hidden_state, hidden_layers).cuda()
        self.regression = nn.Sequential(
            nn.Linear(self.inputlayers, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.regression = self.regression.cuda()
        self.hidden = torch.zeros(hidden_layers, 100, hidden_state)

    def forward(self, x_cat, x_num, x_words):
        current_bs = len(x_cat)
        x_embeds = None
        for i, embeds in enumerate(self.embeddings):
            #x_embeds = torch.cat([x_embeds, embeds(x[:, i].long().view(current_bs, -1)).view(current_bs, -1)], dim=1) if x_embeds is not None else embeds(x[:, i].long().view(current_bs, -1)).view(current_bs, -1)
            x_embeds = torch.cat([ x_embeds, embeds(x_cat[:, i])], dim=1).view(current_bs, -1) if x_embeds is not None else embeds(x_cat[:, i]).view(current_bs, -1)
        x_words = self.embedding_words(x_words)
        x_words, self.hidden = self.gru(x_words)
        x = torch.cat([x_words.view(current_bs, -1) ,x_embeds, x_num], dim=1)
        x = self.dropout1(x)
        x = self.regression(x)
        return x

if __name__ == '__main__':
    # load
    dropcols = ['user_id']
    with timer('Load data'):
        train = pd.read_feather('../features/featured/train_full_5.feather')
        y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])
        train_words = pd.read_csv('../features/train/seq_tokenized.csv')
        train_words['seq_title_description'] = train_words['seq_title_description'].str.strip('[]').str.replace('\s+', '')
        train_words['seq_title_description'] = train_words['seq_title_description'].apply(lambda x: x.split(','))
        train['seq_title_description'] = train_words
        del train_words; gc.collect()
        train.drop(dropcols, axis=1, inplace=True)

    # categorical
    categorical_cols = ['region',
                        'city',
                        'parent_category_name',
                        'category_name',
                        'param_1',
                        'user_type',
                        #'user_id',
                        'image_top_1',
                        'param_2',
                        'param_3',
                        #'item_seq_number'
                        ]
    categorical_dict = {'region': (31, 8),
                        'city'  : (1786, 16),
                        'parent_category_name': (12, 8),
                        'category_name': (49, 8),
                        'param_1': (378, 16),
                        'user_type': (4, 8),
#                        'param_123': (3220, 16),
                        #'user_id': (1009911, 32),
                        'image_top_1': (3066, 16),
                        'param_2': (280, 16),
                        'param_3': (1279, 16),
                        #'item_seq_number': (33949, 32)
                        }
    numeric_cols = [col for col in train.columns if not col in categorical_cols+['seq_title_description']+dropcols]
    columns = train.columns

    # model
    batch_size = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = GRUNet(categorical_dict=categorical_dict, batch_size=batch_size, numeric=numeric_cols, word_embed='./wordembedding/new_cc_matrix.npy')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    # split into train and valid dataset
    test_size = 0.05
    train_x, valid_x, train_y, valid_y = train_test_split(train, y, test_size=test_size)
    del train, y; gc.collect()

    # create dataloader
    with timer('Create Dataloader'):
        train_dataloader = create_dataloader(train_x, train_y, categorical=categorical_cols, numeric=numeric_cols,
                                            batch_size=batch_size, shuffle=True)
        valid_dataloader = create_dataloader(valid_x, valid_y, categorical=categorical_cols, numeric=numeric_cols,
                                            batch_size=batch_size, shuffle=False)

    epoch_num = 1
    print('training')
    for epoch in range(epoch_num):
        scheduler.step()
        with timer(f'Train: {epoch+1}-epoch'):
            # Training
            net.train()
            running_loss = 0
            num_iter = 0
            for iter, data in tqdm(enumerate(train_dataloader, 0)):
                # learning
                inputs_cat, inputs_num, inputs_words, labels = data
                inputs_cat = inputs_cat.to(device)
                inputs_num = inputs_num.to(device)
                inputs_words = inputs_words.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs_cat, inputs_num, inputs_words)
                loss = criterion(outputs, labels)
                loss = torch.sqrt(loss)
                loss.backward()
                optimizer.step()

                # print result
                running_loss += loss.item()
                num_iter += 1
                if (iter+1) % 1000 == 0:
                    print(f'[{(iter+1)*batch_size}, {epoch+1}] loss: {running_loss/num_iter}')

            # validation
            net.eval()
            valid_loss = 0
            num_iter = 0
            with torch.no_grad():
                for iter, data in enumerate(valid_dataloader, 0):
                    inputs_cat, inputs_num, inputs_words, labels = data
                    inputs_cat = inputs_cat.to(device)
                    inputs_num = inputs_num.to(device)
                    inputs_words = inputs_words.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs_cat, inputs_num, inputs_words)
                    loss = criterion(outputs, labels)
                    loss = torch.sqrt(loss)
                    valid_loss += loss.item()
                    num_iter +=1
            print(f'Validation loss: {valid_loss/num_iter}')
    del train_dataloader, valid_dataloader; gc.collect()

    # Submission
    with timer('Submission'):
        test = pd.read_feather('../features/featured/test_full_5.feather')
        test_words = pd.read_csv('../features/test/seq_tokenized.csv')
        test_words['seq_title_description'] = test_words['seq_title_description'].str.strip('[]').str.replace('\s+', '')
        test_words['seq_title_description'] = test_words['seq_title_description'].apply(lambda x: x.split(','))
        test['seq_title_description'] = test_words
        del test_words; gc.collect()
        test.drop(dropcols, axis=1, inplace=True)
        dummy_y = pd.DataFrame(np.zeros(test.shape[0]))
        test_dataloader = create_dataloader(test, y=dummy_y, categorical=categorical_cols, numeric=numeric_cols,
                                            batch_size=batch_size, shuffle=False)

        net.eval()
        preds = None
        with torch.no_grad():
            for iter, data in enumerate(test_dataloader, 0):
                inputs_cat, inputs_num, inputs_words, _ = data
                inputs_cat = inputs_cat.to(device)
                inputs_num = inputs_num.to(device)
                inputs_words = inputs_words.to(device)
                current_bs = len(inputs_cat)
                outputs = net(inputs_cat, inputs_num, inputs_words)
                outputs = outputs.view(current_bs, 1)
                if torch.cuda.is_available():
                    outputs = outputs.cpu().numpy().flatten()
                else:
                    outputs = outputs.numpy().flatten()                    
                preds = np.concatenate([preds, outputs]) if preds is not None else outputs
        preds = np.clip(preds, 0, 1)

        subs = pd.read_csv('../input/test.csv', usecols=['item_id'])
        subs['deal_probability'] = preds

        datetime = dt.now().strftime('%Y_%m%d_%H%M_%S')
        fileprefix = '../subs/NN_GRU_'
        filename = fileprefix+datetime+'.csv.gz'
        subs.to_csv(filename, index=False, compression='gzip')

        print(subs.shape)
        del subs; gc.collect()