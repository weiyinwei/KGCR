from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np
from Metric import rank, full_accuracy

def full_ranking(epoch, model, data, user_item_inter, mask_items, is_training, step, topk, model_name, prefix, writer=None): 
    print(prefix+' start...')
    model.eval()
    with no_grad():               
        if model_name == 'KGCR_new':
            model.infer()
        all_index_of_rank_list = rank(model.num_u, user_item_inter, mask_items, model.result, model.u_result, model.hat_u_result, model.i_result, model.hat_i_result, is_training, step, topk, model_name)
        precision, recall, ndcg_score = full_accuracy(data, all_index_of_rank_list, user_item_inter, is_training, topk)

        print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
            epoch, precision, recall, ndcg_score))
            
        return [precision, recall, ndcg_score]



