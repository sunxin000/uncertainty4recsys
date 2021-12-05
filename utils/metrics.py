import torch
import numpy as np
from torchmetrics.functional import accuracy
def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0

def dcg():
	pass
	

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