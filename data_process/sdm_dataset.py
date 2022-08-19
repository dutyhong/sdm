# 作者 ：duty
# 创建时间 ：2022/8/5 10:36 上午
# 文件 ：sdm_dataset.py
import torch

class SdmDataSet(torch.utils.data.Dataset):
	"""
	SDM模型的训练数据生成器
	target_item_shop_id, target_item_category_id, current_window_shop_ids, current_window_category_ids, history_window_shop_ids, \
		history_window_category_ids, user_id = input_data
	"""
	def __init__(self, datapath, user_map, shop_map, category_map, item_map, max_length):
		file = open(datapath, "r")
		self.user_ids = []
		self.shop_ids = []
		self.category_ids = []
		# self.item_ids = []
		self.current_window_shop_ids_list = []
		self.current_window_category_ids_list = []
		self.history_window_shop_ids_list = []
		self.history_window_category_ids_list = []
		self.targets = []
		for line in file.readlines():
			columns = line.rstrip().split("\t")
			item_id = int(columns[0])
			user_id = columns[1]
			shop_id = columns[2]
			category_id = columns[3]
			history_shop_ids = columns[4].split(",")
			history_category_ids = columns[5].split(",")
			current_window_shop_ids = history_shop_ids[0:5]
			current_window_shop_ids = list(map(lambda x:shop_map.get(x, 0), current_window_shop_ids))
			history_window_shop_ids = history_shop_ids[5:20]
			history_window_shop_ids = list(map(lambda x:shop_map.get(x, 0), history_window_shop_ids))
			current_window_category_ids = history_category_ids[0:5]
			current_window_category_ids = list(map(lambda x: category_map.get(x, 0), current_window_category_ids))
			history_window_category_ids = history_category_ids[5:20]
			history_window_category_ids = list(map(lambda x: category_map.get(x, 0), history_window_category_ids))
			self.user_ids.append(user_map.get(user_id,0))
			self.shop_ids.append(shop_map.get(shop_id,0))
			self.category_ids.append(category_map.get(category_id,0))
			self.current_window_shop_ids_list.append(current_window_shop_ids)
			self.current_window_category_ids_list.append(current_window_category_ids)
			self.history_window_shop_ids_list.append(history_window_shop_ids)
			self.history_window_category_ids_list.append(history_window_category_ids)
			self.targets.append(item_id)

	def __len__(self):
		return len(self.user_ids)

	def __getitem__(self, index):
		user_id = torch.tensor(self.user_ids[index])
		shop_id = torch.tensor(self.shop_ids[index])
		category_id = torch.tensor(self.category_ids[index])
		current_window_shop_ids = torch.tensor(self.current_window_shop_ids_list[index])
		current_window_category_ids = torch.tensor(self.current_window_category_ids_list[index])
		history_window_shop_ids = torch.tensor(self.history_window_shop_ids_list[index])
		history_window_category_ids = torch.tensor(self.history_window_category_ids_list[index])
		target = torch.tensor(self.targets[index])
		return shop_id, category_id,current_window_shop_ids, current_window_category_ids,history_window_shop_ids,history_window_category_ids, user_id, target
