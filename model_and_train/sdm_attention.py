# 作者 ：duty
# 创建时间 ：2022/6/23 6:04 下午
# 文件 ：sdm_attention.py
import torch
from torch import nn


class SdmAttention(nn.Module):
	def __init__(self, embedding_dim):
		super().__init__()
		self.weight_layer = nn.Linear(embedding_dim, 1)

	def forward(self, query, keys):
		"""
		:param query:[B E]
		:param keys: [B T E]
		:return:
		"""
		# [B, E] = query.shpe
		[B, T, E] = keys.shape
		device = query.device
		query = torch.ones((B, T, 1)).to(device)*query.view(B,1,E) ## 将query行向量复制T遍
		att_in = torch.cat([query,keys], dim=2)
		att_out = self.weight_layer(att_in) ## [B T 1]
		att_out = att_out.squeeze(2)
		softmax_weight = torch.softmax(att_out, dim=1) ## [B T]
		softmax_weight = softmax_weight.view(B, 1, T)
		out = torch.matmul(softmax_weight, keys) ## 加权求和
		return out

