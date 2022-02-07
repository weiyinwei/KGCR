import torch
import math 
import torch.nn.functional as F
import torch.nn as nn

def rank(num_user, user_item_inter, mask_items, result, u_result, hat_u_result, i_result, hat_i_result, is_training, step, topk, model_name):
    user_tensor = result[:num_user]
    item_tensor = result[num_user:]
    user_rep = u_result
    hat_user_rep = hat_u_result #F.normalize(hat_u_result)
    item_rep = i_result
    hat_item_rep = hat_i_result# F.normalize(hat_i_result)
    print(model_name)

    start_index = 0
    end_index = num_user if step==None else step
    all_index_of_rank_list = torch.LongTensor([])

    while end_index <= num_user and start_index < end_index:
        temp_user_tensor = user_tensor[start_index:end_index]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

        if model_name == 'DKGR':
            # temp_user_rep = user_rep[start_index:end_index] # len*dim
            # u_i_e_score = torch.matmul(temp_user_rep, item_rep.t())#len*num_item
            # #ex_uie_score = torch.matmul(user_rep.mean(0), item_rep.t()) u* * i
            # # ex_uie_score = torch.mean(u_i_e_score, dim=1, keepdim=True)#len*1
            # ex_uie_score = torch.mean(u_i_e_score, dim=0, keepdim=True)#1*num_item
            
            # temp_hat_user_rep = hat_user_rep[start_index:end_index]#len*dim
            # temp_score = torch.matmul(temp_hat_user_rep, item_rep.t())#len*num_item

            # temp_score = (u_i_e_score-ex_uie_score)*torch.sigmoid(temp_score)#len*num_item * (len*num_item)
            # # temp_score = (u_i_e_score-ex_uie_score)*(temp_score)#len*num_item * (len*num_item)
            
            # score_matrix += temp_score

            temp_user_rep = user_rep[start_index:end_index] # len*dim
            u_i_e_score = torch.matmul(temp_user_rep, item_rep.t())#len*num_item
            ex_uie_score = torch.mean(u_i_e_score, dim=0, keepdim=True)#1*num_item
            
            temp_hat_user_rep = hat_user_rep[start_index:end_index]#len*dim
            temp_score = torch.matmul(temp_hat_user_rep, hat_item_rep.t())#len*num_item
            temp_score = (u_i_e_score-ex_uie_score)*torch.sigmoid(temp_score)#len*num_item * (len*num_item)
            
            score_matrix += temp_score      


        elif model_name == 'DKGR_new_new' or model_name == 'DKGR_new' or model_name == 'KGCR' or model_name == 'KGCR_new' :
            temp_user_rep = user_rep[start_index:end_index] # len*dim
            u_i_e_score = torch.matmul(temp_user_rep, item_rep.t())#len*num_item
            ex_uie_score = torch.mean(u_i_e_score, dim=0, keepdim=True)#1*num_item
            
            temp_hat_user_rep = hat_user_rep[start_index:end_index]#len*dim
            temp_score = torch.matmul(temp_hat_user_rep, hat_item_rep.t())#len*num_item
            temp_score = (u_i_e_score-ex_uie_score)*torch.sigmoid(temp_score)#len*num_item * (len*num_item)
            
            score_matrix += temp_score      

        else:
            temp_user_rep = user_rep[start_index:end_index]
            u_i_e_score = torch.matmul(temp_user_rep, item_rep.t())
            score_matrix += u_i_e_score
        
        if is_training is False:
            for row, col in user_item_inter.items():

                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-num_user
                    score_matrix[row][col] = 1e-15

        _, index_of_rank_list = torch.topk(score_matrix, topk)
        all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+num_user), dim=0)

        start_index = end_index
        if end_index+step < num_user:
            end_index += step
        else:
            end_index = num_user

    return all_index_of_rank_list


def full_accuracy(val_data, all_index_of_rank_list, user_item_inter, is_training, topk):
    length = 0      
    precision = recall = ndcg = 0.0

    # if is_training:
    #     for row, col in user_item_inter.items():
    #         user = row
    #         pos_items = set(col)
    #         num_pos = len(pos_items)
    #         if num_pos == 0:
    #             continue
    #         length += 1
    #         items_list = all_index_of_rank_list[user].tolist()
    #         items = set(items_list)
    #         num_hit = len(pos_items.intersection(items))
    #         precision += float(num_hit / topk)
    #         recall += float(num_hit / num_pos)
    #         ndcg_score = 0.0
    #         max_ndcg_score = 0.0
    #         for i in range(min(num_hit, topk)):
    #             max_ndcg_score += 1 / math.log2(i+2)
    #         if max_ndcg_score == 0:
    #             continue
    #         for i, temp_item in enumerate(items_list):
    #             if temp_item in pos_items:
    #                 ndcg_score += 1 / math.log2(i+2)
    #         ndcg += ndcg_score/max_ndcg_score
    # else:
    sum_num_hit = 0
    for data in val_data:
        user = data[0]
        pos_items = set(data[1:])
        num_pos = len(pos_items)
        if num_pos == 0:
            continue
        length += 1
        items_list = all_index_of_rank_list[user].tolist()
        items = set(items_list)

        num_hit = len(pos_items.intersection(items))
        sum_num_hit += num_hit
        precision += float(num_hit / topk)
        recall += float(num_hit / num_pos)
        ndcg_score = 0.0
        max_ndcg_score = 0.0
        for i in range(min(num_pos, topk)):
            max_ndcg_score += 1 / math.log2(i+2)

        if max_ndcg_score == 0:
            continue
        for i, temp_item in enumerate(items_list):
            if temp_item in pos_items:
                ndcg_score += 1 / math.log2(i+2)
        ndcg += ndcg_score/max_ndcg_score

    return precision/length, recall/length, ndcg/length