import numpy as np
import torch
from torchmetrics.functional import accuracy
from sklearn.metrics import roc_auc_score


def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0

def recall_at_k(label, score, user_num, items_per_user, eval=True, dataset='coat', user_ids=None):
	if dataset == 'kuairand':
		label = label.numpy() if torch.is_tensor(label) else label
		score = score.numpy() if torch.is_tensor(score) else score
		user_ids = user_ids.numpy() if torch.is_tensor(user_ids) else user_ids
		
		user_ids_unique = np.unique(user_ids)
		ks = [2, 4, 6] if eval else [1, 2, 3, 4, 5, 6]
		recalls = {k: [] for k in ks}    
		valid_users = 0  # 添加有效用户计数
		
		for user_id in user_ids_unique:
			index = np.where(user_ids == user_id)[0]  # 确保获取索引数组
			label_user = label[index]
			score_user = score[index]
			convert_sum = np.sum(label_user)
			
			# 跳过无正样本的用户，但不返回0
			if convert_sum == 0:
				continue
				
			valid_users += 1  # 增加有效用户计数
			index_user = np.argsort(-score_user)
			sorted_labels = label_user[index_user]

			for k in ks:
				if k <= len(sorted_labels):  # 确保k不超过可用项目数
					recall_k = np.sum(sorted_labels[:k]) / convert_sum  # 归一化
					recalls[k].append(recall_k)
		
		# 只在有有效用户时计算平均值
		average_recalls = np.array([np.mean(recalls[k]) if recalls[k] else 0.0 for k in ks])
		return average_recalls
	else:
		score = np.array(score)
		label = np.array(label)
		convert = label.reshape([user_num, items_per_user])
		score = score.reshape([user_num, items_per_user])
		convert_sum = np.sum(convert, axis=1)
		index = np.where(convert_sum > 0)
		convert = convert[index]
		score = score[index]
		user_num = convert.shape[0]
		###
		index = np.argsort(-score)  # Descending order
		convert = np.array([convert[i][index[i]] for i in range(user_num)])
		recall_k = []
		if eval:
			for k in [2, 4, 6]:
				convert_k = convert[:, :k]
				recall_k.append(np.sum(convert_k) / user_num)
		else:
			for k in [1, 2, 3, 4, 5, 6]:
				convert_k = convert[:, :k]
				recall_k.append(np.sum(convert_k) / user_num)
		return np.array(recall_k)


def dcg_at_k(label, score, user_num, items_per_user, eval=True, dataset='coat', user_ids=None):
	if dataset == 'kuairand':
		label = label.numpy() if torch.is_tensor(label) else label
		score = score.numpy() if torch.is_tensor(score) else score
		user_ids = user_ids.numpy() if torch.is_tensor(user_ids) else user_ids
		
		user_ids_unique = np.unique(user_ids)
		ks = [2, 4, 6] if eval else [1, 2, 3, 4, 5, 6]
		dcgs = {k: [] for k in ks}    
		
		for user_id in user_ids_unique:
			index = np.where(user_ids == user_id)[0]
			label_user = label[index]
			score_user = score[index]
			
			# 检查用户是否有足够的交互记录和至少一个正样本
			if len(label_user) == 0 or np.sum(label_user) == 0:
				continue
				
			index_user = np.argsort(-score_user)
			sorted_labels = label_user[index_user]

			for k in ks:
				if k <= len(sorted_labels):
					order = np.log2(np.arange(k) + 2)
					convert_k = sorted_labels[:k]
					dcg = np.sum(np.divide(convert_k, order))
					if not np.isnan(dcg):
						dcgs[k].append(dcg)
		
		average_dcgs = np.array([np.mean(dcgs[k]) if dcgs[k] else 0.0 for k in ks])
		return average_dcgs
		

	else:
		score = np.array(score)
		label = np.array(label)

		convert = label.reshape([user_num, items_per_user])
		score = score.reshape([user_num, items_per_user])
		###
		convert_sum = np.sum(convert, axis=1)
		index = np.where(convert_sum > 0)
		convert = convert[index]
		score = score[index]
		user_num = convert.shape[0]
		###
		index = np.argsort(-score)  # Descending order
		convert = np.array([convert[i][index[i]] for i in range(user_num)])
		dcg_k = []
		if eval:
			for k in [2, 4, 6]:
				convert_k = convert[:, :k]
				order = np.log2(np.arange(k) + 2)
				dcg = np.divide(convert_k, order)
				dcg_k.append(np.sum(dcg) / user_num)
		else:
			for k in [1, 2, 3, 4, 5, 6]:
				convert_k = convert[:, :k]
				order = np.log2(np.arange(k) + 2)
				dcg = np.divide(convert_k, order)
				dcg_k.append(np.sum(dcg) / user_num)
		return np.array(dcg_k)

	

def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0



def metrics(model, test_loader, top_k, device):
	HR, NDCG, PRECISION= [], [], [] 

	for user, item, label in test_loader:
		user = user.to(device)
		item = item.to(device)

		predictions = model(user, item)
		PRECISION.append(accuracy(predictions.cpu(), label, task='binary').numpy())

		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		ng_item = item[0].item() # leave one-out evaluation has only one item per user
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))

	return np.mean(HR), np.mean(NDCG), 	np.mean(PRECISION)

def calculate_auc(label, score, user_num, items_per_user, dataset='coat', user_ids=None):
	label = np.array(label)
	score = np.array(score)
	
	if dataset == 'kuairand':
		user_ids_unique = np.unique(user_ids)
		aucs = []
		for user_id in user_ids_unique:
			index = np.where(user_ids == user_id)
			label_user = label[index]
			score_user = score[index]
			# 只在用户有正负样本时计算AUC
			if len(np.unique(label_user)) > 1:
				user_auc = roc_auc_score(label_user, score_user)
				aucs.append(user_auc)
		return np.mean(aucs) if aucs else 0.0
	else:
		# 重塑为用户-物品矩阵
		label = label.reshape([user_num, items_per_user])
		score = score.reshape([user_num, items_per_user])
		
		# 计算每个用户的AUC
		user_aucs = []
		for user_label, user_score in zip(label, score):
			if len(np.unique(user_label)) > 1:  # 只在用户有正负样本时计算AUC
				user_auc = roc_auc_score(user_label, user_score)
				user_aucs.append(user_auc)
		
		return np.mean(user_aucs) if user_aucs else 0.0