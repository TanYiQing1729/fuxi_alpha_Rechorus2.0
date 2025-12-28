# -*- coding: UTF-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	# x: (..., D)
	return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _relative_position_bucket(relative_positions: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
	"""T5-style log bucket for non-negative relative distances.

	relative_positions: any shape, expected >= 0
	returns: same shape, int64 bucket ids in [0, num_buckets-1]
	"""
	rel = relative_positions.clamp(min=0).to(torch.long)
	if num_buckets <= 1:
		return torch.zeros_like(rel)

	max_exact = num_buckets // 2
	is_small = rel < max_exact

	rel_if_large = rel.clamp(min=max_exact)
	# avoid log(0)
	rel_if_large_f = rel_if_large.to(torch.float32)
	log_scale = math.log(max_distance / max_exact) if max_distance > max_exact else 1.0
	bucket_if_large = max_exact + (
		torch.log(rel_if_large_f / max_exact) / log_scale * (num_buckets - max_exact)
	).to(torch.long)
	bucket_if_large = bucket_if_large.clamp(max=num_buckets - 1)

	return torch.where(is_small, rel, bucket_if_large)


class _MultiChannelAttention(nn.Module):
	def __init__(
		self,
		emb_size: int,
		num_heads: int,
		max_len: int,
		num_time_buckets: int,
		time_bucket_max_distance: int,
		dropout: float,
		attn_dropout: float,
	):
		super().__init__()
		assert emb_size % num_heads == 0
		self.emb_size = emb_size
		self.num_heads = num_heads
		self.head_dim = emb_size // num_heads
		self.max_len = max_len
		self.num_time_buckets = num_time_buckets
		self.time_bucket_max_distance = time_bucket_max_distance

		self.q_proj = nn.Linear(emb_size, emb_size)
		self.k_proj = nn.Linear(emb_size, emb_size)
		self.v_proj = nn.Linear(emb_size, emb_size)

		# Learnable relative bias tables
		self.rel_pos_bias = nn.Embedding(2 * max_len - 1, 1)
		self.rel_time_bias = nn.Embedding(num_time_buckets, 1)

		self.out_proj = nn.Linear(emb_size * 3, emb_size)
		self.dropout = nn.Dropout(dropout)
		self.attn_dropout = nn.Dropout(attn_dropout)

	def _build_rel_pos_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
		# (L, L) index in [0, 2L-2]
		pos = torch.arange(seq_len, device=device)
		rel = pos[None, :] - pos[:, None] + (seq_len - 1)
		rel = rel.clamp(min=0, max=2 * seq_len - 2)
		# (L, L, 1) -> (1, 1, L, L)
		bias = self.rel_pos_bias(rel).permute(2, 0, 1).unsqueeze(0)
		return bias

	def _build_rel_time_bias(self, timestamps: torch.Tensor, seq_len: int) -> torch.Tensor:
		# timestamps: (B, L)
		# dt(q,k) = ts_q - ts_k, non-negative for causal use
		dt = (timestamps.unsqueeze(2) - timestamps.unsqueeze(1)).clamp(min=0)
		bucket = _relative_position_bucket(dt, self.num_time_buckets, self.time_bucket_max_distance)
		bias = self.rel_time_bias(bucket).squeeze(-1).unsqueeze(1)  # (B, 1, L, L)
		return bias

	def forward(
		self,
		x: torch.Tensor,
		valid: torch.Tensor,
		timestamps: torch.Tensor,
		use_pos: bool,
		use_time: bool,
		use_latent: bool,
	) -> torch.Tensor:
		# x: (B, L, D)
		# valid: (B, L) bool
		B, L, D = x.shape
		device = x.device

		timestamps = timestamps.to(torch.long)

		q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,dh)
		k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
		v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

		# (B,H,L,L)
		scores_latent = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
		scores_pos = self._build_rel_pos_bias(L, device).expand(B, self.num_heads, L, L)
		scores_time = self._build_rel_time_bias(timestamps, L).expand(B, self.num_heads, L, L)

		# masks
		causal = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool)).view(1, 1, L, L)
		key_valid = valid.view(B, 1, 1, L)
		query_valid = valid.view(B, 1, L, 1)
		mask = causal & key_valid & query_valid

		neg_inf = torch.finfo(scores_latent.dtype).min
		scores_latent = scores_latent.masked_fill(~mask, neg_inf)
		scores_pos = scores_pos.masked_fill(~mask, neg_inf)
		scores_time = scores_time.masked_fill(~mask, neg_inf)

		# NOTE: 消融时可能关闭某些通道。为保持 shape 恒定，这里用全 0 输出占位。
		zeros = torch.zeros((B, self.num_heads, L, self.head_dim), device=device, dtype=v.dtype)
		if use_latent:
			attn_latent = self.attn_dropout(F.softmax(scores_latent, dim=-1))
			o_latent = torch.matmul(attn_latent, v)
		else:
			o_latent = zeros
		if use_pos:
			attn_pos = self.attn_dropout(F.softmax(scores_pos, dim=-1))
			o_pos = torch.matmul(attn_pos, v)
		else:
			o_pos = zeros
		if use_time:
			attn_time = self.attn_dropout(F.softmax(scores_time, dim=-1))
			o_time = torch.matmul(attn_time, v)
		else:
			o_time = zeros

		# (B,H,L,3*dh) -> (B,L,3*D)
		o = torch.cat([o_pos, o_time, o_latent], dim=-1).transpose(1, 2).contiguous().view(B, L, 3 * D)
		o = self.out_proj(o)
		o = self.dropout(o)
		return o


class _StandardSelfAttention(nn.Module):
	def __init__(self, emb_size: int, num_heads: int, dropout: float, attn_dropout: float):
		super().__init__()
		assert emb_size % num_heads == 0
		self.emb_size = emb_size
		self.num_heads = num_heads
		self.head_dim = emb_size // num_heads
		self.q_proj = nn.Linear(emb_size, emb_size)
		self.k_proj = nn.Linear(emb_size, emb_size)
		self.v_proj = nn.Linear(emb_size, emb_size)
		self.out_proj = nn.Linear(emb_size, emb_size)
		self.dropout = nn.Dropout(dropout)
		self.attn_dropout = nn.Dropout(attn_dropout)

	def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
		# x: (B, L, D)
		# valid: (B, L) bool
		B, L, D = x.shape
		device = x.device

		q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,dh)
		k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
		v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,L,L)

		causal = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool)).view(1, 1, L, L)
		key_valid = valid.view(B, 1, 1, L)
		query_valid = valid.view(B, 1, L, 1)
		mask = causal & key_valid & query_valid

		neg_inf = torch.finfo(scores.dtype).min
		scores = scores.masked_fill(~mask, neg_inf)
		attn = self.attn_dropout(F.softmax(scores, dim=-1))
		o = torch.matmul(attn, v)  # (B,H,L,dh)
		o = o.transpose(1, 2).contiguous().view(B, L, D)
		o = self.out_proj(o)
		o = self.dropout(o)
		return o


class _MSFFN(nn.Module):
	def __init__(self, emb_size: int, hidden_size: int, dropout: float, single_stage: bool = False, eps: float = 1e-6):
		super().__init__()
		self.single_stage = single_stage
		self.dropout = dropout
		self.eps = eps

		self.lin0 = nn.Linear(emb_size, emb_size)
		if not single_stage:
			self.lin1 = nn.Linear(emb_size, hidden_size, bias=False)
			self.lin2 = nn.Linear(hidden_size, emb_size, bias=False)
			self.lin3 = nn.Linear(emb_size, hidden_size, bias=False)

	def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
		x = self.lin0(F.dropout(x, p=self.dropout, training=self.training)) + x0
		if self.single_stage:
			return x
		normed = _rms_norm(x, eps=self.eps)
		normed = F.dropout(normed, p=self.dropout, training=self.training)
		x1 = F.silu(self.lin1(normed)) * self.lin3(normed)
		x = self.lin2(x1) + x
		return x


class _FuXiAlphaLayer(nn.Module):
	def __init__(
		self,
		emb_size: int,
		num_heads: int,
		max_len: int,
		num_time_buckets: int,
		time_bucket_max_distance: int,
		hidden_size: int,
		dropout: float,
		attn_dropout: float,
		use_ams: bool,
		use_pos: bool,
		use_time: bool,
		use_latent: bool,
		ffn_single_stage: bool,
	):
		super().__init__()
		self.use_ams = use_ams
		self.use_pos = use_pos
		self.use_time = use_time
		self.use_latent = use_latent
		if use_ams:
			self.attn = _MultiChannelAttention(
				emb_size=emb_size,
				num_heads=num_heads,
				max_len=max_len,
				num_time_buckets=num_time_buckets,
				time_bucket_max_distance=time_bucket_max_distance,
				dropout=dropout,
				attn_dropout=attn_dropout,
			)
		else:
			self.attn = _StandardSelfAttention(
				emb_size=emb_size,
				num_heads=num_heads,
				dropout=dropout,
				attn_dropout=attn_dropout,
			)
		self.ffn = _MSFFN(
			emb_size=emb_size,
			hidden_size=hidden_size,
			dropout=dropout,
			single_stage=ffn_single_stage,
		)

	def forward(self, x: torch.Tensor, valid: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
		x0 = x
		if self.use_ams:
			a = self.attn(
				x,
				valid=valid,
				timestamps=timestamps,
				use_pos=self.use_pos,
				use_time=self.use_time,
				use_latent=self.use_latent,
			)
		else:
			a = self.attn(x, valid=valid)
		x = a + x0
		x = self.ffn(x, x0=x)
		return x


class FuXiAlpha(SequentialModel):
	"""FuXi-α (简化的 dense 版，适配 ReChorus SeqReader)。

	- 输入: history_items, history_times, lengths
	- 输出: 对候选 item_id 的打分 (pos + neg)

	说明：原仓库使用 jagged tensor + fbgemm op；这里实现等价的 padded-seq 版本，保证 Windows/CPU 可跑。
	"""

	reader = 'SeqReader'
	runner = 'BaseRunner'
	extra_log_args = [
		'emb_size',
		'num_layers',
		'num_heads',
		'fuxi_use_ams',
		'fuxi_use_pos',
		'fuxi_use_time',
		'fuxi_use_latent',
		'fuxi_single_stage_ffn',
		'time_buckets',
		'time_bucket_max',
		'ff_hidden_size',
		'attn_dropout',
	]

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64, help='Embedding size.')
		parser.add_argument('--num_layers', type=int, default=1, help='Number of FuXi blocks.')
		parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
		parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout.')
		parser.add_argument('--ff_hidden_size', type=int, default=256, help='FFN hidden size.')
		parser.add_argument('--fuxi_use_ams', type=int, default=1, help='Use AMS (multi-channel) attention; set 0 to use standard self-attention (1/0).')
		parser.add_argument('--fuxi_use_pos', type=int, default=1, help='Use positional channel in multi-channel attention (1/0).')
		parser.add_argument('--fuxi_use_time', type=int, default=1, help='Use time channel in multi-channel attention (1/0).')
		parser.add_argument('--fuxi_use_latent', type=int, default=1, help='Use latent (qk) channel in multi-channel attention (1/0).')
		parser.add_argument('--fuxi_single_stage_ffn', type=int, default=0, help='Use single-stage FFN (ablation for MSFFN stage-2) (1/0).')
		parser.add_argument('--time_buckets', type=int, default=32, help='Number of time buckets.')
		parser.add_argument('--time_bucket_max', type=int, default=1000000, help='Max distance for log time bucket.')
		return SequentialModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.emb_size = args.emb_size
		self.num_layers = args.num_layers
		self.num_heads = args.num_heads
		self.attn_dropout = args.attn_dropout
		self.ff_hidden_size = args.ff_hidden_size
		self.fuxi_use_ams = int(getattr(args, 'fuxi_use_ams', 1))
		self.fuxi_use_pos = int(args.fuxi_use_pos)
		self.fuxi_use_time = int(args.fuxi_use_time)
		self.fuxi_use_latent = int(args.fuxi_use_latent)
		self.fuxi_single_stage_ffn = int(args.fuxi_single_stage_ffn)
		self.time_buckets = args.time_buckets
		self.time_bucket_max = args.time_bucket_max
		self.max_his = args.history_max

		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
		self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

		self.layers = nn.ModuleList([
			_FuXiAlphaLayer(
				emb_size=self.emb_size,
				num_heads=self.num_heads,
				max_len=self.max_his,
				num_time_buckets=self.time_buckets,
				time_bucket_max_distance=self.time_bucket_max,
				hidden_size=self.ff_hidden_size,
				dropout=self.dropout,
				attn_dropout=self.attn_dropout,
				use_ams=bool(self.fuxi_use_ams),
				use_pos=bool(self.fuxi_use_pos),
				use_time=bool(self.fuxi_use_time),
				use_latent=bool(self.fuxi_use_latent),
				ffn_single_stage=bool(self.fuxi_single_stage_ffn),
			)
			for _ in range(self.num_layers)
		])

		# precompute [0..max_his-1] on device in forward

	def forward(self, feed_dict):
		self.check_list = []
		i_ids = feed_dict['item_id']  # (B, C)
		history = feed_dict['history_items']  # (B, L)
		timestamps = feed_dict['history_times']  # (B, L)
		lengths = feed_dict['lengths']  # (B,)

		B, L = history.shape
		device = history.device

		valid = history.gt(0)
		his = self.i_embeddings(history)

		# Position embedding (same indexing style as SASRec: most recent has larger index)
		len_range = torch.arange(L, device=device)
		position = (lengths[:, None] - len_range[None, :]) * valid.long()
		position = position.clamp(min=0, max=self.max_his)
		his = his + self.p_embeddings(position)

		# layers
		x = his
		for layer in self.layers:
			x = layer(x, valid=valid, timestamps=timestamps)

		x = x * valid[:, :, None].float()
		last_idx = (lengths - 1).clamp(min=0)
		u_vec = x[torch.arange(B, device=device), last_idx, :]  # (B, D)

		i_vec = self.i_embeddings(i_ids)  # (B, C, D)
		prediction = (u_vec[:, None, :] * i_vec).sum(-1)
		return {'prediction': prediction.view(B, -1)}
