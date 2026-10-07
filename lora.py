"""低秩适配：原线性层冻住，只训练旁边的小矩阵，结束后加回原权重。"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """前向是原线性层加 scale * x @ A^T @ B^T。A 用小随机数，B 从 0 开始，所以一开始等于原层。"""

    def __init__(self, base, rank, alpha):
        super().__init__()
        if base.bias is not None:
            raise ValueError("当前线性层都没有 bias")
        self.base = base
        self.scale = alpha / rank
        for param in self.base.parameters():
            param.requires_grad_(False)
        # 和原权重同一块设备，否则 MPS 上的前向会对不上。
        device = base.weight.device
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=device) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)) * self.scale


def apply_lora(model, rank, alpha):
    """给注意力和前馈里的线性层挂上 LoRA。词嵌入、输出层和打分头不动。"""

    def visit(module):
        for name, child in list(module.named_children()):
            if name in ("lm_head", "value_head"):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
            else:
                visit(child)

    visit(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad_(True)
            module.B.requires_grad_(True)


def merge_lora(model):
    """把 BA 加进原权重，再换回普通线性层，后面的阶段继续训完整模型。"""

    def visit(module):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                merged = nn.Linear(child.base.in_features, child.base.out_features, bias=False)
                merged = merged.to(child.base.weight.device)
                with torch.no_grad():
                    merged.weight.copy_(child.base.weight + child.scale * (child.B @ child.A))
                setattr(module, name, merged)
            else:
                visit(child)

    visit(model)
    for param in model.parameters():
        param.requires_grad_(True)
