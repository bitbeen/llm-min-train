# 最小的大模型训练

[中文](#中文) · [English](#english)

一套从随机参数写起的解码器训练。分词、网络、损失、工具调用和评估都在这个仓库里，不加载别人的权重。逐段概念在 [`docs/concepts.md`](docs/concepts.md)。

A decoder trained from random initialization. The tokenizer, network, losses, tool loop, and evaluation all live in this repo. Nothing is loaded from someone else's checkpoint. Stage-by-stage notes are in [`docs/concepts.md`](docs/concepts.md).

## 中文

这份代码用来把一条现代语言模型的训练链路摊开看：因果语言模型、监督微调、偏好对比、组内采样、低秩适配、混合专家、视觉块，以及「模型写调用、程序填结果」的工具循环。宽度大约 **0.75B**（24 层，隐层 1536，24 头，前馈 4096）。精确参数量在 `python train.py` 开头打印。语料是手写的中文样本加上 WikiText-2 的英文，规模只够把这些机制跑通，不能拿去和几千亿参数的模型比通用能力。

### 模型

`TinyLM` 是一个只看左边的解码器。每一层是 Pre-Norm：RMSNorm，然后残差。

RMSNorm 不减均值，只按最后一维的均方根缩放，再乘可学习的增益。自注意力是多头的。查询和键在送进点积之前做 RoPE：把头维度对半切开，按位置旋转。频率是 \(10000^{-i/(\mathrm{head\_dim}/2)}\)，正余弦按 `max_seq` 预先算好，同一设备上复用。因果掩码是上三角，填充位置另有 `key_padding_mask`。

普通层的前馈是 SwiGLU：一条支路过 SiLU，和另一条相乘，再投回 `d_model`。第 2 层（`moe_layer = 1`）换成 4 个同样的专家。路由是一个没有偏置的线性层，对专家做 softmax，只计算 top-2，再把这两个门控重新归一化。负载损失用的是 Switch 那一种：

\[
\mathcal{L}_{\mathrm{aux}} = N \sum_i \mathrm{importance}_i \cdot \mathrm{dispatch}_i
\]

`importance` 是门控概率在 batch 上的平均，`dispatch` 是该专家被选进 top-k 的比例。某一路被反复选中时这一项变大，系数是 `moe_aux = 0.01`。没被选中的专家这一步没有梯度，裁剪时会跳过。

词嵌入和输出层绑同一份矩阵。预测下一个子词就是在词表上做分类。旁边有一个 `value_head`，读每个位置的隐状态，给整段回答打一个标量，组内采样用它当基线。

图不另建编码器。32×32 的图按 8×8 不重叠切块，卷积把每块映射成 `d_model` 维，得到 16 个向量，接在文字前面。这些位置的标签是 -100，交叉熵不算它们。视频是两帧，形状 `(B, 帧, 3, H, W)`，两帧的块沿时间拼成更长的前缀。左右分色的样本用来迫使模型使用块的位置，而不是只记住「这张图是什么颜色」这句话。

### 损失

位置 \(t\) 的 logits 预测位置 \(t+1\)。`causal_loss` 把标签左移一位，`ignore_index=-100`。指令样本里，问题部分全是 -100，损失只落在回答和结尾符上。总损失是交叉熵加上 \(\mathcal{L}_{\mathrm{aux}}\)。优化器是 AdamW，梯度范数裁到 1。

监督微调之后还有三段偏好学习：

- **DPO。** 同一问题的好回答和坏回答。参考模型是这一步开始时的拷贝，全程冻住。记策略和参考在回答上的对数概率为 \(\pi\) 和 \(\pi_{\mathrm{ref}}\)，损失是

\[
-\log\sigma\Big(\beta\big((\pi_c-\pi_{\mathrm{ref},c})-(\pi_r-\pi_{\mathrm{ref},r})\big)\Big)
\]

  \(\beta\) 是 `dpo_beta`。间隔拉得越大，损失越小。

- **GRPO。** 默认关掉（`grpo_steps = 0`）。打开后，对同一道改短题按温度采样一组回答。奖励是和标准答案的字重合：整句相同为 1，否则 \(0.7\) 的召回加 \(0.3\) 的前缀重合。优势是奖励减去 `value_head` 的分数，打分头再用均方误差去拟合这个奖励。策略梯度只落在回答的 token 上。

- **LoRA。** 注意力和前馈里的线性层冻住，旁边加 \(\frac{\alpha}{r} x A^{\top} B^{\top}\)。\(A\) 用标准差 0.02 的随机数，\(B\) 从 0 开始，所以第一步等于原层。秩 8，\(\alpha = 16\)。学完把低秩项加回原权重，后面的看图和工具继续训完整模型。词嵌入和输出层绑在一起，这两处不加 LoRA，否则新字的向量会被冻住。

### 工具和推理

算术不在网络里算。`act` 生成到 `</工具>` 就停，解析 `<工具>乘:7:8</工具>` 这种调用，程序做整数加减乘，把 `<结果>56</结果>` 接回上下文，模型再决定要不要调下一次。最多两次。连算 `2加3减1` 因此是两步：先得到 5，再用 5 去减。

`reason` 把这个循环教成固定格式：先写 `<思考>先取7乘8</思考>`，再写工具，看见程序填的结果之后只复读那个数字。思考几个字如果词表里没有，会补在末尾并加长嵌入矩阵，旧字的向量原样拷过来。若干格子不进训练，用来看模型是在抄算式，还是在背见过的得数。

`tasks` 把三件事放进同一步：中文改短原题、删掉英文虚词之后更短的 Wiki 句子、1 到 9 的加减乘，以及视频动作名。改短算对的标准是回答更短，并且用字来自原句。

样本很少，所以每步把它们重复拼到大约 32–96 条再前向。序列只有几十个 token 时，单条前向在 5090 上核函数还没铺开就返回了。32 位矩阵走 TF32。

### 训练顺序

1. WikiText-2 上预测下一个子词。
2. `samples/pretrain.txt` 上继续预测下一个子词。
3. 监督微调：`sft.jsonl` 的改短，`dialogue.jsonl` 的多轮。多轮损失只算最后一轮回答。
4. GRPO，默认跳过。
5. DPO。
6. LoRA，学 `lora.jsonl`，然后合并。
7. CIFAR-10（约 400 张，类别名是中文）和 `mm.jsonl` 的纯色、左右分色。
8. UCF101 子集，约 24 段，每段两帧，回答是英文动作名。
9. `tool.jsonl` 的单步和两步调用。
10. 文字、适配句子和看图再走一段，用较小的学习率收尾。
11. `main()` 结束时调用 `reason`。

看图放在工具前面：图像是新输入，一开始梯度大。等它稳住再学工具，回放时不容易把调用格式冲掉。

### 跑起来

Python 3.10 及以上。有 CUDA 用 CUDA，否则 MPS，再否则 CPU。RTX 5090（sm_120）需要 CUDA 13 的 PyTorch 2.11，版本钉在 `requirements.txt`。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hf_data.py
python train.py
python generate.py
```

已经有 `out/model.pt` 时：

```bash
python train.py reason   # 先取数，写工具，读结果
python train.py tasks    # 中英文改短、加减乘、视频
python eval.py out       # 九九乘法、500 段改短、50 张图片
```

`data/hf/` 和 `out/` 在 `.gitignore` 里。仓库不带下载的语料，也不带权重。

| 文件 | 作用 |
| --- | --- |
| `model.py` | 解码器、RoPE、MoE、与词嵌入绑在一起的输出层 |
| `train.py` | 各阶段的损失和更新 |
| `data.py` | 批次。问题位置标成 -100 |
| `tokenizer.py` | 128 轮 BPE。没见过的字会报错 |
| `tools.py` | 生成、解析调用、把结果接回去 |
| `vision.py` | 切块，以及纯色、左右分色 |
| `lora.py` | 低秩旁路，结束后合并 |
| `eval.py` | 乘法表、改短、图片 |

## English

This repo is a place to read a modern language-model training stack in one directory: a causal language model, supervised fine-tuning, preference comparison, grouped sampling, low-rank adaptation, a mixture of experts, vision patches, and a tool loop where the model writes the call and the program fills in the result. The width is about **0.75B** parameters (24 layers, hidden size 1536, 24 heads, feed-forward 4096). The exact count is printed at the start of `python train.py`. The data is a handful of handwritten Chinese samples plus English WikiText-2. That is enough to exercise the mechanisms. It is not a general model at the scale of hundreds of billions of parameters.

### Model

`TinyLM` is a left-to-right decoder. Each block is pre-norm: RMSNorm, then a residual.

RMSNorm does not subtract the mean. It rescales by the root-mean-square over the last dimension and multiplies by a learned gain. Attention is multi-head. Queries and keys are rotated with RoPE before the dot product: split the head dimension in half and rotate by position. Frequencies are \(10000^{-i/(\mathrm{head\_dim}/2)}\). Cosine and sine are cached for `max_seq` on the current device. A triangular mask blocks future positions. `key_padding_mask` blocks padding.

The ordinary feed-forward is SwiGLU: one branch goes through SiLU, multiplies the other branch, and projects back to `d_model`. Layer 2 (`moe_layer = 1`) replaces that with 4 experts of the same shape. The router is a bias-free linear map. It softmaxes over experts, runs only the top 2, and renormalizes those two gates. The load-balancing term is the Switch auxiliary loss:

\[
\mathcal{L}_{\mathrm{aux}} = N \sum_i \mathrm{importance}_i \cdot \mathrm{dispatch}_i
\]

`importance` is the batch-mean gate probability. `dispatch` is the fraction of tokens that put that expert in the top-k. The term grows when one expert is chosen over and over. It is added with `moe_aux = 0.01`. An expert that was not selected has no gradient on that step, and the clip skips it.

Token embeddings and the output layer share one matrix, so next-subword prediction is a classification over the vocabulary. A `value_head` reads each hidden state and scores the whole answer. Grouped sampling uses that score as the baseline.

Images do not get a separate encoder. A 32×32 image is cut into non-overlapping 8×8 patches. A convolution maps each patch to `d_model`, which is 16 vectors, and those vectors are prepended to the text. Their labels are -100, so the cross-entropy ignores them. Video is two frames, shaped `(B, frames, 3, H, W)`. The patch sequences are concatenated in time. Split-color images force the model to use where a patch sits, rather than memorize the question "what color is this picture?".

### Losses

Logits at position \(t\) predict the token at \(t+1\). `causal_loss` shifts the labels by one and uses `ignore_index=-100`. On instruction rows the prompt is entirely -100, so the loss sits on the answer and the end token. The total loss is the cross-entropy plus \(\mathcal{L}_{\mathrm{aux}}\). The optimizer is AdamW. Gradients are clipped to norm 1.

After supervised fine-tuning there are three preference stages:

- **DPO.** A chosen answer and a rejected answer for the same prompt. The reference is a frozen copy from the start of the stage. With policy and reference log-probabilities \(\pi\) and \(\pi_{\mathrm{ref}}\) on the answer tokens, the loss is

\[
-\log\sigma\Big(\beta\big((\pi_c-\pi_{\mathrm{ref},c})-(\pi_r-\pi_{\mathrm{ref},r})\big)\Big)
\]

  \(\beta\) is `dpo_beta`. A wider margin makes the loss smaller.

- **GRPO.** Off by default (`grpo_steps = 0`). When enabled, several answers are sampled for one shorten prompt. The reward is character overlap with the label: 1 for an exact string, otherwise \(0.7\) recall plus \(0.3\) prefix overlap. The advantage is the reward minus the `value_head` score, and the value head is trained with mean squared error against that reward. The policy gradient is only on answer tokens.

- **LoRA.** Linear layers inside attention and the feed-forward stay frozen. A side path adds \(\frac{\alpha}{r} x A^{\top} B^{\top}\). \(A\) is random with standard deviation 0.02 and \(B\) starts at zero, so the first step matches the original layer. Rank is 8 and \(\alpha\) is 16. After training, the low-rank term is merged back and later vision and tool stages train the full model. Embeddings and the output layer are tied, so they do not get LoRA. Freezing them would freeze vectors for any new character.

### Tools and reasoning

Arithmetic is not computed inside the network. `act` stops at `</工具>`, parses a call such as `<工具>乘:7:8</工具>`, runs integer add, subtract, or multiply, appends `<结果>56</结果>`, and lets the model decide whether to call again. The cap is two calls. The chain `2加3减1` is therefore two steps: produce 5, then subtract from 5.

`reason` teaches that loop as a fixed format. The model first writes `<思考>先取7乘8</思考>`, then the tool call, and after the program inserts the result it only repeats that number. If 思考, 先, or 取 are missing from the vocabulary, they are appended and the embedding matrix grows. The old rows are copied unchanged. Some cells are held out of training, to see whether the model copies the equation or memorizes a product it has already seen.

`tasks` puts three jobs in one step: the Chinese shorten labels, English Wiki sentences shortened by dropping function words, add/subtract/multiply from 1 to 9, and video action names. A shorten answer counts when it is shorter than the source and its characters come from that source.

The sample set is small, so each step repeats and packs it to about 32–96 sequences before the forward. A single short sequence returns from the kernel before an RTX 5090 is busy. Float32 matmuls run in TF32.

### Order

1. Next-subword prediction on WikiText-2.
2. The same objective on `samples/pretrain.txt`.
3. Supervised fine-tuning: shorten rows in `sft.jsonl`, multi-turn rows in `dialogue.jsonl`. Multi-turn loss is only on the last answer.
4. GRPO, skipped by default.
5. DPO.
6. LoRA on `lora.jsonl`, then a merge.
7. CIFAR-10 (about 400 images, Chinese class names) plus solid and split colors in `mm.jsonl`.
8. A UCF101 subset, about 24 clips, two frames each, English action names.
9. One-step and two-step calls in `tool.jsonl`.
10. Another pass over text, the LoRA sentences, and images, at a lower learning rate.
11. `main()` ends by calling `reason`.

Vision comes before tools because images are a new input and the early gradients are large. Tool format is learned after that settles, so replaying vision is less likely to erase the call syntax.

### Run

Python 3.10 or newer. The device is CUDA, then MPS, then CPU. An RTX 5090 (sm_120) needs PyTorch 2.11 built with CUDA 13. That pin is in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hf_data.py
python train.py
python generate.py
```

When `out/model.pt` already exists:

```bash
python train.py reason   # read the numbers, write the tool, repeat the result
python train.py tasks    # Chinese and English shorten, arithmetic, video
python eval.py out       # multiplication table, 500 shorten passages, 50 images
```

`data/hf/` and `out/` are gitignored. The repo does not carry the downloaded corpora or the weights.

| File | Role |
| --- | --- |
| `model.py` | Decoder, RoPE, MoE, output layer tied to the token embedding |
| `train.py` | Losses and updates for each stage |
| `data.py` | Batches. Prompt positions are labeled -100 |
| `tokenizer.py` | 128 rounds of BPE. An unseen character raises |
| `tools.py` | Generate, parse a call, append the result |
| `vision.py` | Patches, solid colors, split colors |
| `lora.py` | Low-rank side path, merged at the end |
| `eval.py` | Multiplication table, shorten, images |
