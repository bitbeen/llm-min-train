from dataclasses import dataclass


@dataclass
class Config:
    """宽度、上下文和每个阶段的步数。全程只有这一份模型。"""

    # 这一组宽度大约是 0.75B。d_model 必须能被 n_heads 整除，头维度还要是偶数，RoPE 才好对半旋转。
    d_model: int = 1536
    n_layers: int = 24
    n_heads: int = 24
    d_ff: int = 4096
    max_seq: int = 128
    image_size: int = 32
    patch_size: int = 8
    dropout: float = 0.0
    n_experts: int = 4
    top_k: int = 2
    # 哪一层把前馈换成专家。-1 表示这一层仍用普通前馈。
    moe_layer: int = 1
    # 专家负载不均时加到交叉熵上的系数。
    moe_aux: float = 0.01
    lr: float = 3e-4
    pretrain_steps: int = 120
    sft_steps: int = 160
    dpo_steps: int = 16
    # 采样阶段在句子还没记牢时会把回答带成重复字，先关掉。
    grpo_steps: int = 0
    grpo_group: int = 4
    # 大于 1 时同一问题的几条采样才会散开，组内优势才不是 0。
    grpo_temperature: float = 1.1
    # 采样回答偏离参考模型时的惩罚。
    grpo_beta: float = 0.04
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_steps: int = 120
    tool_steps: int = 80
    mm_steps: int = 160
    # 工具单独学完后，再用文字和看图拉回来，最后一起收尾。
    tool_only_steps: int = 160
    keep_steps: int = 100
    tool_star_steps: int = 80
    polish_steps: int = 200
    # 文本只走一小段。图片每类只留两张，步数够把类别名记下来。
    hf_text_steps: int = 40
    hf_image_steps: int = 320
    hf_video_steps: int = 80
    # 子词合并轮数。每一轮把语料里出现至少两次的最高频相邻对收成一个词。
    bpe_merges: int = 128
    # 打分头去拟合奖励时，这项损失的系数。
    value_coef: float = 0.5
    dpo_beta: float = 0.1
    seed: int = 7
