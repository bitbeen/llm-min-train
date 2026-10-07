import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from vision import PatchEmbed


class RMSNorm(nn.Module):
    """按最后一维的均方根缩放，再乘可学习的 weight。不减均值。"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * scale


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)。前半和后半旋转，让注意力知道位置。
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """多头自注意力。查询和键带旋转位置，掩码同时挡住未来的字和填充。"""

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, key_padding_mask):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, cos[:t], sin[:t])
        k = apply_rope(k, cos[:t], sin[:t])

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 上三角挡住未来的位置，每个字只能看自己和左边。
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:
            # 填充位置不参与注意力，避免补齐用的 <pad> 被当成内容。
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.proj(out)


class SwiGLU(nn.Module):
    """两条支路：一条过 SiLU，和另一条相乘，再投回 d_model。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoE(nn.Module):
    """一组 SwiGLU 专家。每个 token 只把概率最高的 top_k 个专家算出来，再按门控加权。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg) for _ in range(cfg.n_experts))

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        gates = F.softmax(self.router(flat), dim=-1)
        top_val, top_idx = torch.topk(gates, self.top_k, dim=-1)
        top_val = top_val / top_val.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # 负载均衡：专家被选中的比例，和门控平均概率，两者差太大会加进损失。
        importance = gates.mean(dim=0)
        dispatch = torch.zeros_like(gates).scatter(1, top_idx, 1.0).mean(dim=0)
        aux = self.n_experts * (importance * dispatch).sum()

        out = flat.new_zeros(flat.shape)
        for expert_id, expert in enumerate(self.experts):
            picked = (top_idx == expert_id).any(dim=-1)
            if not bool(picked.any()):
                continue
            weight = (top_val * (top_idx == expert_id).float()).sum(dim=-1)
            index = picked.nonzero(as_tuple=False).flatten()
            values = expert(flat.index_select(0, index)) * weight.index_select(0, index).unsqueeze(-1)
            out = out.index_add(0, index, values)
        return out.view(b, t, d), aux


class Block(nn.Module):
    """一层解码器：先注意力再前馈，都带残差。use_moe 时前馈换成专家。"""

    def __init__(self, cfg: Config, use_moe=False):
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MoE(cfg) if use_moe else SwiGLU(cfg)

    def forward(self, x, cos, sin, key_padding_mask):
        x = x + self.attn(self.norm1(x), cos, sin, key_padding_mask)
        hidden = self.norm2(x)
        if self.use_moe:
            y, aux = self.mlp(hidden)
            return x + y, aux
        # 普通前馈没有负载项，第二项保持 0，调用方可以和专家层一样相加。
        return x + self.mlp(hidden), x.new_zeros(())

class TinyLM(nn.Module):
    """解码器语言模型。传入 images 时，把图像块向量接在文字前面。"""

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.vision = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(cfg, use_moe=(layer_idx == cfg.moe_layer)) for layer_idx in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # 输出层和词嵌入共用权重，预测下一个字就是在词表上做分类。
        self.lm_head.weight = self.tok_emb.weight
        # 给整段回答打一个分，训练时去拟合这条回答拿到的奖励。
        self.value_head = nn.Linear(cfg.d_model, 1, bias=False)
        self._rope = None

    def rope(self, device):
        """每个位置一对 cos/sin。换设备时重算，同一设备上复用。"""
        if self._rope is None or self._rope[0].device != device:
            half = self.cfg.d_model // self.cfg.n_heads // 2
            freq = 1.0 / (10000 ** (torch.arange(half, device=device).float() / half))
            pos = torch.arange(self.cfg.max_seq, device=device).float()
            angles = torch.outer(pos, freq)
            self._rope = (angles.cos(), angles.sin())
        return self._rope

    def forward(self, input_ids, images=None, key_padding_mask=None):
        x = self.tok_emb(input_ids)
        # 图像块接在文字前面，这些位置没有字，填充掩码标成有效。
        # 视频是 (B, 帧数, 3, H, W)，每一帧各切一块再顺着时间接上。
        if images is not None:
            if images.dim() == 5:
                frames = [self.vision(images[:, frame]) for frame in range(images.size(1))]
                image_tokens = torch.cat(frames, dim=1)
            else:
                image_tokens = self.vision(images)
            x = torch.cat((image_tokens, x), dim=1)
            if key_padding_mask is not None:
                image_pad = torch.zeros(
                    key_padding_mask.size(0),
                    image_tokens.size(1),
                    dtype=torch.bool,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat((image_pad, key_padding_mask), dim=1)
        if x.size(1) > self.cfg.max_seq:
            raise ValueError("序列长度 %d 超过 max_seq %d" % (x.size(1), self.cfg.max_seq))
        cos, sin = self.rope(x.device)
        aux = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x, cos, sin, key_padding_mask)
            aux = aux + block_aux
        hidden = self.norm(x)
        # 每个位置一个分数。调用方取回答最后一个位置，当作模型对这条回答的估计。
        return self.lm_head(hidden), aux, self.value_head(hidden).squeeze(-1)


def causal_loss(logits, labels):
    """位置 t 的 logits 预测位置 t+1。标签为 -100 的位置不计入损失。"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
