# new_kg_list.npy: <h, r, t>
# train_list.npy: <u, i>
# test_dict.npy: <u, i>
# user_item_dict.npy: <u,{i}> train&test
# user_item_dict_train.npy: <u,{i}> train
# user_entity_dict.npy: <u, {e}> train
# u_e_list.npy: <u, e> train
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def data_load(dataset):
    if dataset == 'yelp2018':
        num_a = 90961
        num_u = 45919
        num_i = 45538
        num_r = 42
    elif dataset == 'lastfm':
        num_a = 58266
        num_u = 23566
        num_i = 48123
        num_r = 9
    elif dataset == 'amazon-book':
        num_a = 88572
        num_u = 70679
        num_i = 24915
        num_r = 39

    dir_str = './datasets/' + dataset
    kg_data = np.load(dir_str+'/new_kg_list.npy')
    train_data = np.load(dir_str+'/train_list.npy')
    test_data = np.load(dir_str+'/test_dict.npy', allow_pickle=True)
    user_item_dict = np.load(dir_str+'/user_item_dict.npy', allow_pickle=True).item()
    user_item_dict_train = np.load(dir_str+'/user_item_dict_train.npy', allow_pickle=True).item()
    h_r_dict = np.load(dir_str+'/h_r_dict.npy', allow_pickle=True).item()
    # u_e_list = np.load(dir_str+'/u_e_list.npy')
    print('11111')
    u_e_list = np.load(dir_str+'/u_e_list_new.npy')

    kg_list = np.column_stack((kg_data[:,0], kg_data[:,2]))
    relation_list = kg_data[:,1]
    u_e_index = np.load(dir_str+'/all_u_a_list.npy').tolist()
    u_e_value = np.load(dir_str+'/all_value_list.npy').tolist()
    att_weight = torch.sparse_coo_tensor(u_e_index, u_e_value, (num_u, num_a))
    print(num_a, num_u, num_i)


    return train_data, test_data, kg_list, relation_list, u_e_list, user_item_dict, user_item_dict_train, h_r_dict, num_u, num_i, num_r, num_a, att_weight


class TrainDataset(Dataset):
    def __init__(self, train_data, user_item_dict, num_i, num_u):
        self.train_data = train_data
        self.user_item_dict = user_item_dict
        self.all_item = set(range(num_u, num_u+num_i))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, item = self.train_data[index]
        while True:
            neg_item = random.sample(self.all_item, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break

        return torch.LongTensor([user,user]), torch.LongTensor([item,neg_item])

class KGDataset(Dataset):
    def __init__(self, kg_data, relation_list, h_r_dict, num_e, num_r):
        self.kg_data = kg_data
        self.relation_list = relation_list
        self.h_r_dict = h_r_dict
        self.num_e = num_e

    def __len__(self):
        return len(self.kg_data)

    def __getitem__(self, index):
        h, t = self.kg_data[index]
        r = self.relation_list[index]

        while True:
            neg_t = random.randint(0, self.num_e-1)

            if neg_t != h and neg_t not in self.h_r_dict[(h, r)]:
                break

        return torch.LongTensor([h, r]), torch.LongTensor([t, neg_t])

class AKGDataset(Dataset):
    def __init__(self, kg_data, relation_list, h_r_dict, num_e, num_r):
        self.kg_data = kg_data
        self.relation_list = relation_list
        self.h_r_dict = h_r_dict
        self.num_e = num_e

    def __len__(self):
        return len(self.kg_data)

    def __getitem__(self, index):
        h, t = self.kg_data[index]
        r = self.relation_list[index]
        neg_t = random.randint(0, self.num_e-1)
        if neg_t == h or neg_t in self.h_r_dict[(h, r)]:
            neg_t = self.num_e

        return torch.LongTensor([h, r]), torch.LongTensor([t, neg_t])


if __name__ == '__main__':
    train_data, val_data, val_label, test_data, test_label, user_att_dict, item_att_dict, user_item_dict, all_item, att_num, user_num, item_num = data_load('LON')
    # train_dataset = MyDataset(train_data, user_att_dict, item_att_dict, all_item, user_item_dict)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10)
    # val_dataset = VTDataset(val_data, val_label, user_att_dict, item_att_dict)
    # val_dataloader = DataLoader(val_dataset, batch_size=51)
    # for a, u, i, l in val_dataloader:
    #     print(a.size(), u.size(), i.size(), l.size())
    #     print(l)
 

