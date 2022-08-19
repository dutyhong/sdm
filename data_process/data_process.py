# 作者 ：duty
# 创建时间 ：2022/6/13 7:49 下午
# 文件 ：data_process.py
import pandas as pd
import pickle

def train_test_samples():
	data = pd.read_csv("/home/tizi/tz_nlp/data/rec/din_click_samples.csv", nrows=20000)
	# file = open("/home/tizi/tz_nlp/data/rec/din_user_map", "w")
	# file2 = open("/home/tizi/tz_nlp/data/rec/din_shop_map", "w")
	# file3 = open("/home/tizi/tz_nlp/data/rec/din_category_map", "w")
	file4 = open("part_din_train_samples", "w")
	file5 = open("part_din_test_samples", "w")
	user_map = set()
	shop_map = set()
	category_map = set()
	item_map = set()
	for index, row in data.iterrows():
		print(index)
		if row["click_label"]!=1:
			continue
		user_uuid = str(row["user_uuid"])
		user_map.add(user_uuid)
		click_label = row["click_label"]
		shop_id = str(row["shop_id"])
		shop_map.add(shop_id)
		cat_back_l3 = str(row["cat_back_l3"])
		category_map.add(cat_back_l3)
		item_id = str(row["item_id"])
		item_map.add(item_id)
		history_shop_ids = row["history_shop_ids"]
		history_shop_map = {}
		history_shop_ids = history_shop_ids.split(";")
		for history_shop_id in history_shop_ids:
			hshop_id = history_shop_id.split(",")[0]
			shop_map.add(hshop_id)
			history_shop_map[int(history_shop_id.split(",")[1])] = hshop_id
		history_cat_back_l3s = row["history_cat_back_l3s"]
		history_category_map = {}
		history_cat_back_l3s = history_cat_back_l3s.split(";")
		for history_cat_back_l3 in history_cat_back_l3s:
			hcat_back_l3 = history_cat_back_l3.split(",")[0]
			category_map.add(hcat_back_l3)
			history_category_map[int(history_cat_back_l3.split(",")[1])] = hcat_back_l3
		##将历史搜索店铺和类目按照顺序写入训练样本
		sorted_history_shop_map = sorted(history_shop_map.items(), key=lambda item:item[0], reverse=False)
		sorted_history_category_map = sorted(history_category_map.items(), key=lambda item: item[0], reverse=False)
		sorted_history_shop_list = []
		sorted_history_category_list = []
		for e in sorted_history_shop_map:
			shop_id = e[1]
			sorted_history_shop_list.append(shop_id)
		for e in sorted_history_category_map:
			category_id = e[1]
			sorted_history_category_list.append(category_id)
		if index<18000:
			file4.write(str(item_id)+"\t"+user_uuid+"\t"+str(shop_id)+"\t"+str(cat_back_l3)+"\t"+",".join(sorted_history_shop_list)+"\t"+",".join(sorted_history_category_list)+"\n")
		else:
			file5.write(str(item_id) + "\t" + user_uuid + "\t" + str(shop_id) + "\t" + str(cat_back_l3) + "\t" + ",".join(sorted_history_shop_list) + "\t" + ",".join(sorted_history_category_list) + "\n")
	file4.close()
	file5.close()
	user_dict = {}
	category_dict = {}
	shop_dict = {}
	item_dict = {}
	for i, e in enumerate(user_map):
		user_dict[e] = i
	for i, e in enumerate(shop_map):
		shop_dict[e] = i
	for i, e in enumerate(category_map):
		category_dict[e] = i
	for i, e in enumerate(item_map):
		item_dict[e] = i
	pickle.dump(user_dict,open("part_user_dict.pkl", "wb"))
	pickle.dump(shop_dict,open("part_shop_dict.pkl", "wb"))
	pickle.dump(category_dict,open("part_category_dict.pkl", "wb"))
	pickle.dump(item_dict, open("part_item_dict.pkl", "wb"))

def samples_analysis():
	file1 = open("../../../../data/rec/din_train_samples", "r")
	file2 = open("../../../../data/rec/din_test_samples", "r")
	train_positive_cnt = 0
	train_negtive_cnt = 0
	test_positive_cnt = 0
	test_negtive_cnt = 0
	for line in file1.readlines():
		columns = line.rstrip().split("\t")
		label = int(columns[0])
		if label==1:
			train_positive_cnt = train_positive_cnt+1
		else:
			train_negtive_cnt = train_negtive_cnt+1
	for line in file2.readlines():
		columns = line.rstrip().split("\t")
		label = int(columns[0])
		if label==1:
			test_positive_cnt = test_positive_cnt+1
		else:
			test_negtive_cnt = test_negtive_cnt+1
	print("train label positive %d,negtive %d\n"%(train_positive_cnt, train_negtive_cnt))
	print("test label positive %d,negtive %d\n"%(test_positive_cnt, test_negtive_cnt))
##train label positive 773232,negtive 1026768

##test label positive 89649,negtive 110351
if __name__=="__main__":
	# samples_analysis()
	train_test_samples()
	print("ddd")
