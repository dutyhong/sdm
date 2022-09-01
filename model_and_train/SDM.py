# 作者 ：duty
# 创建时间 ：2022/6/23 3:35 下午
# 文件 ：SDIN.py
import torch
from torch import nn


## 参考阿里SDM的模型结构， 当前窗口为短期兴趣，历史窗口为长期兴趣；当前窗口通过多头注意力机制提取用户的多个兴趣，通过selfattention过滤掉无关的行为
## 参考链接 https://blog.csdn.net/wuzhongqiang/article/details/123856954
from model_and_train.sdm_attention import SdmAttention
from model_and_train.sampled_softmax_loss import SampledSoftmax

class TzSdm(nn.Module):
	def __init__(self, user_num, shop_num, category_num,item_num, current_window_seq_length, history_window_seq_length, embedding_dim, lstm_hidden_size,categorical_feature_num):
		super().__init__()
		self.user_embed_layer = nn.Embedding(user_num, embedding_dim)
		# self.target_shop_embed_layer = nn.Embedding(shop_num, embedding_dim)
		# self.target_category_embed_layer = nn.Embedding(category_num, embedding_dim)
		# self.current_window_shop_embed_layer = nn.Embedding(shop_num, embedding_dim)
		# self.current_window_category_embed_layer = nn.Embedding(category_num, embedding_dim)
		# self.history_window_shop_embed_layer = nn.Embedding(shop_num, embedding_dim)
		# self.history_window_category_embed_layer = nn.Embedding(category_num, embedding_dim)
		## 是分开每个单独embedding还是只用一个实例化embedding 那个效果好 ？？？？？
		self.shop_embed_layer = nn.Embedding(shop_num, embedding_dim)
		self.category_embed_layer = nn.Embedding(category_num, embedding_dim)
		self.multi_head_attention_layer = nn.MultiheadAttention(embedding_dim, 2)
		self.shop_lstm_layer = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, batch_first=True, num_layers=2)
		self.category_lstm_layer = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, batch_first=True,num_layers=2)
		self.self_attention = SdmAttention(embedding_dim*2)
		self.history_dense_layer = nn.Linear(embedding_dim*categorical_feature_num, embedding_dim)
		self.gate_dense_layer = nn.Linear(embedding_dim*3, embedding_dim)
		self.sampled_softmax = SampledSoftmax(item_num,nhid=embedding_dim, nsampled=200,tied_weight=None)

	def forward(self, input_data):
		target_item_shop_id, target_item_category_id, current_window_shop_ids, current_window_category_ids, history_window_shop_ids, \
		history_window_category_ids, user_id = input_data
		user_embedding = self.user_embed_layer(user_id)
		## 短期用户行为结构
		current_window_shop_embedding = self.shop_embed_layer(current_window_shop_ids)
		current_window_category_embedding = self.category_embed_layer(current_window_category_ids)
		history_window_shop_embedding = self.shop_embed_layer(history_window_shop_ids)
		history_window_category_embedding = self.category_embed_layer(history_window_category_ids)
		## 当前的窗口先经过lstm， 然后多头注意力，然后selfattention；每个lstm的输入为商品的某一类特征：比如shop的序列，category的序列等
		current_shop_lstm_out = self.shop_lstm_layer(current_window_shop_embedding)[0]
		##输出output,(hn,cn), output是最后一层每个时间步的输出， 所以取output[:,-1,:], hn为每个时间步最后一层的输出
		current_category_lstm_out = self.category_lstm_layer(current_window_category_embedding)[0]
		mha_in = torch.cat([current_shop_lstm_out, current_category_lstm_out], dim=1)
		mha_out = self.multi_head_attention_layer(mha_in,mha_in, mha_in)[0] ## Q K V都一样， 输出为[B T E], 返回att和att_weights
		## 然后跟user做selfattention
		self_att_out = self.self_attention(user_embedding, mha_out) ## [B 1 E]
		short_out = torch.squeeze(self_att_out, 1)
		## 长期用户行为结构，根据每个类别特征同user做attention最后concat
		shop_att_out = torch.squeeze(self.self_attention(user_embedding, history_window_shop_embedding),1)
		category_att_out = torch.squeeze(self.self_attention(user_embedding, history_window_category_embedding), 1)
		history_cat_in = torch.cat([shop_att_out, category_att_out], dim=1)
		long_out = self.history_dense_layer(history_cat_in)

		##长短期兴趣融合
		long_short_cat_in = torch.cat([short_out, long_out, user_embedding], dim=1)
		long_short_cat_out = self.gate_dense_layer(long_short_cat_in) ## [B E]
		gate_values = torch.sigmoid(long_short_cat_out)
		negtive_gate_values = torch.ones_like(gate_values)-gate_values
		long_short_out = gate_values*short_out+negtive_gate_values*long_out
		target_shop_embedding = self.shop_embed_layer(target_item_shop_id)
		target_category_embedding = self.category_embed_layer(target_item_category_id)
		## 如果是计算相似度
		target_embedding = torch.add(target_shop_embedding,target_category_embedding)
		# 简化了损失函数的计算，直接余弦距离
		cos_out = torch.cosine_similarity(long_short_out, target_embedding)
		return torch.sigmoid(cos_out)
		# return long_short_out
