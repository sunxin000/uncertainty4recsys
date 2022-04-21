import torch
import numpy as np
from torchmetrics.functional import accuracy
def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0

def dcg_at_k(label, score, user_num, items_per_user ,eval=True):
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
		PRECISION.append(accuracy(predictions.cpu(), label).numpy())

		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		ng_item = item[0].item() # leave one-out evaluation has only one item per user
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))

	return np.mean(HR), np.mean(NDCG), 	np.mean(PRECISION)