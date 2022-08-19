# 作者 ：duty
# 创建时间 ：2022/6/24 2:48 下午
# 文件 ：sdm_train.py
import torch
from sklearn.metrics import roc_auc_score
import os
import pickle

from torch.optim import Adam

from data_process.sdm_dataset import SdmDataSet
from model_and_train.SDM import TzSdm


def eval_output(scores, target, loss_function=torch.nn.functional.binary_cross_entropy):
	loss = loss_function(scores, target)

	y_pred = scores.round()
	accuracy = (y_pred == target).type(torch.FloatTensor).mean()

	auc = roc_auc_score(target.cpu().detach(), scores.cpu().detach())
	return loss, accuracy, auc

# train_file = os.path.join( '/home/tizi/tz_nlp/data/rec', "din_train_samples")
# test_file  = os.path.join( '/home/tizi/tz_nlp/data/rec', "din_test_samples")
# uid_voc = os.path.join( '/home/tizi/tz_nlp/data/rec', "user_dict.pkl")
# mid_voc = os.path.join( '/home/tizi/tz_nlp/data/rec', "shop_dict.pkl")
# cat_voc = os.path.join( '/home/tizi/tz_nlp/data/rec', "category_dict.pkl")
train_file =  "part_din_train_samples"
test_file  =  "part_din_test_samples"
uid_voc    =  "part_user_dict.pkl"
mid_voc    =  "part_shop_dict.pkl"
cat_voc    =  "part_category_dict.pkl"
user_map = pickle.load( open( uid_voc, 'rb'))
n_uid = len( user_map)
material_map = pickle.load( open( mid_voc, 'rb'))
n_mid = len( material_map)
category_map = pickle.load( open( cat_voc, 'rb'))
n_cat = len( category_map)
item_map = pickle.load(open("part_item_dict.pkl"), "rb")
n_item = len(item_map)
BATCH_SIZE = 256
MAX_LEN = 20
dataset_train = SdmDataSet( train_file, user_map, material_map, category_map,item_map, max_length = MAX_LEN)
dataset_test = SdmDataSet( test_file, user_map, material_map, category_map,item_map, max_length = MAX_LEN)
print("数据集处理完成！！！")
loader_train = torch.utils.data.DataLoader( dataset_train, batch_size = BATCH_SIZE, shuffle = True)
loader_test = torch.utils.data.DataLoader( dataset_test, batch_size = BATCH_SIZE, shuffle = False)
embedding_dim = 128
lstm_hidden_size = 128
current_window_seq_length = 5
history_window_seq_length = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TzSdm(n_uid, n_mid, n_cat,current_window_seq_length, history_window_seq_length, embedding_dim, lstm_hidden_size, 2).to(device)
optimizer = Adam(model.parameters(),lr=0.001)
loss_func = torch.nn.functional.binary_cross_entropy
EPOCH_TIME = 5
TEST_ITER = 2000
iter = 0
model.train()
for epoch in range(EPOCH_TIME):
	for i, data in enumerate(loader_train):
		iter = iter+1
		data = [item.to(device) for item in data]
		target = data.pop(-1)
		# target.to(device)
		optimizer.zero_grad()

		out = model(data)
		out = torch.squeeze(out)
		loss = loss_func(out.to(device), target.type(torch.FloatTensor).to(device))
		loss.backward()
		optimizer.step()
		loss, accuracy, auc = eval_output(out, target.type(torch.FloatTensor).to(device))
		print("\r[%d/%d][%d/%d]\tloss:%.5f\tacc:%.5f\tauc:%.5f" % (
		epoch + 1, EPOCH_TIME, i + 1, len(loader_train), loss.item(), accuracy.item(), auc.item()), end='')

		if iter % TEST_ITER == 0:
			model.eval()
			with torch.no_grad():
				score_list = []
				target_list = []
				for data in loader_test:
					data = [item.to(device) for item in data]

					target = data.pop(-1)

					scores = model(data)
					score_list.append(torch.squeeze(scores))
					target_list.append(target)
				scores = torch.cat(score_list, dim=-1)
				target = torch.cat(target_list, dim=-1)
				loss, accuracy, auc = eval_output(scores, target.type(torch.FloatTensor).to(device))
				print("\tTest Set\tloss:%.5f\tacc:%.5f\tauc:%.5f" % (loss.item(), accuracy.item(), auc.item()))
			model.train()
	torch.save(model, "sdm_model")
	torch.save(model.state_dict(), "sdm_model_params")