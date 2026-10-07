"""从零训练：预训练、监督微调、组内采样、偏好对比、低秩适配、看图、工具调用。"""

import copy
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config
from data import (
    DATA,
    HF,
    _load_rgb,
    _compress,
    all_texts,
    dpo_pairs,
    grpo_rows,
    hf_image_batches,
    hf_text_batches,
    hf_video_batches,
    lora_batches,
    mm_batches,
    photo_contrast_batch,
    photo_focus_batch,
    pretrain_batches,
    read_jsonl,
    reason_batches,
    reason_example,
    shorten_batches,
    sft_batches,
    text_hold_batch,
    tool_all_echo,
    tool_batches,
    tool_branch_batch,
    tool_call_batch,
    tool_chain_batch,
    tool_echo_contrast,
    tool_fix_calls,
    tool_op_batches,
)
from generate import generate, pick_device, place
from lora import apply_lora, merge_lora
from model import MoE, TinyLM, causal_loss
from tokenizer import BPETokenizer
from tools import act, run_tool
from vision import solid_image, split_image

# 权重写在同目录下的 out/。
OUT = Path(__file__).resolve().parent / "out"


def move_batch(batch, device):
    """批次里的张量跟模型放在同一块设备上。"""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


# 5090 上一条一条前向，核还没铺开就返回了。每步大约这么多条序列。
GPU_BATCH = 96


def fuse_batches(batches, device, target=GPU_BATCH):
    """把一条一条的小批次拼成大批次，放到显卡上。样本不够就重复，让矩阵乘法填满算力。"""
    if not batches:
        return []
    groups = []
    current = []
    signature = None
    for batch in batches:
        image_shape = tuple(batch["images"].shape[1:]) if "images" in batch else ()
        keys = tuple(sorted(key for key, value in batch.items() if torch.is_tensor(value)))
        sig = (keys, image_shape)
        if sig != signature:
            if current:
                groups.append(current)
            current = []
            signature = sig
        current.append(batch)
    if current:
        groups.append(current)
    fused = []
    for group in groups:
        fused.extend(_fuse_same(group, device, target))
    return fused


def _fuse_same(batches, device, target):
    rows = []
    for batch in batches:
        count = batch["input_ids"].size(0)
        for index in range(count):
            row = {
                "input_ids": batch["input_ids"][index],
                "labels": batch["labels"][index],
                "key_padding_mask": batch["key_padding_mask"][index],
            }
            if "images" in batch:
                row["images"] = batch["images"][index]
            rows.append(row)
    if not rows:
        return []
    copies = max(1, (target + len(rows) - 1) // len(rows))
    rows = rows * copies
    fused = []
    for start in range(0, len(rows), target):
        chunk = rows[start : start + target]
        # 样本本来就不够、靠重复补满时，丢掉最后那几条重复的。样本多于一批时，尾批也留下。
        if copies > 1 and len(chunk) < target // 2 and fused:
            continue
        fused.append(_pad_rows(chunk, device))
    return fused or [_pad_rows(rows[:target], device)]


def _pad_rows(rows, device):
    id_width = max(int(row["input_ids"].size(0)) for row in rows)
    label_width = max(int(row["labels"].size(0)) for row in rows)
    count = len(rows)
    input_ids = torch.zeros(count, id_width, dtype=torch.long, device=device)
    labels = torch.full((count, label_width), -100, dtype=torch.long, device=device)
    mask = torch.ones(count, id_width, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        id_len = int(row["input_ids"].size(0))
        lab_len = int(row["labels"].size(0))
        input_ids[index, :id_len] = row["input_ids"].to(device)
        labels[index, :lab_len] = row["labels"].to(device)
        mask[index, :id_len] = row["key_padding_mask"].to(device)
    packed = {"input_ids": input_ids, "labels": labels, "key_padding_mask": mask}
    if "images" in rows[0]:
        packed["images"] = torch.stack([row["images"] for row in rows]).to(device)
    return packed


def trainable(model):
    """LoRA 阶段大部分权重被冻住，优化器只收还开着梯度的参数。"""
    return [param for param in model.parameters() if param.requires_grad]


def clip_step(loss, optimizer, model):
    """反传、裁剪梯度范数、更新。没被选中的专家这一步没有梯度，裁剪时跳过。"""
    optimizer.zero_grad()
    loss.backward()
    apply_grads(optimizer, model)


def apply_grads(optimizer, model):
    """梯度已经累加好时才更新。几组样本要一起学时，不能在中途把梯度清掉。"""
    grads = [param for param in trainable(model) if param.grad is not None]
    if grads:
        torch.nn.utils.clip_grad_norm_(grads, 1.0)
    optimizer.step()


def probe(model, tokenizer, cfg):
    """用固定的几条问题对答案。返回对上的条数，和一行摘要。"""
    checks = []
    for row in read_jsonl(DATA / "sft.jsonl")[:4]:
        checks.append((row["answer"], act(model, tokenizer, row["prompt"])))
    question = "这张图是什么颜色？\n回答："
    for color, answer in (("红", "红色"), ("绿", "绿色"), ("蓝", "蓝色")):
        image = solid_image(color, cfg.image_size)
        checks.append((answer, act(model, tokenizer, question, image=image)))
    sided = split_image("红", "蓝", cfg.image_size)
    checks.append(("红色", act(model, tokenizer, "这张图左边是什么颜色？\n回答：", image=sided)))
    checks.append(("蓝色", act(model, tokenizer, "这张图右边是什么颜色？\n回答：", image=sided)))
    seen = set()
    image_path = HF / "images.jsonl"
    if image_path.exists():
        for row in read_jsonl(image_path):
            if row["answer"] in seen:
                continue
            seen.add(row["answer"])
            photo = _load_rgb(HF / row["file"], cfg.image_size)
            checks.append((row["answer"], act(model, tokenizer, row["prompt"], image=photo)))
            if len(seen) >= 5:
                break
    checks.append(("<工具>加:2:3</工具><结果>5</结果>5", act(model, tokenizer, "算式2加3。\n回答：")))
    checks.append(("<工具>乘:2:3</工具><结果>6</结果>6", act(model, tokenizer, "算式2乘3。\n回答：")))
    score = sum(1 for gold, pred in checks if pred == gold)
    head = " ".join(pred[:8].replace("\n", "") for _, pred in checks[:4])
    tail = " ".join(pred[:16].replace("\n", "") for _, pred in checks[-2:])
    return score, head + " || " + tail


def run_lm(model, batches, steps, lr, title, aux_coef):
    """下一个字的交叉熵，加上专家负载损失。批次比步数少时循环使用。"""
    optimizer = torch.optim.AdamW(trainable(model), lr=lr)
    model.train()
    device = next(model.parameters()).device
    batches = fuse_batches(batches, device)
    shape = batches[0]["input_ids"].shape
    print("%s  batch %d x %d" % (title, shape[0], shape[1]))
    for step in range(1, steps + 1):
        batch = batches[(step - 1) % len(batches)]
        logits, aux, _ = model(
            batch["input_ids"],
            images=batch.get("images"),
            key_padding_mask=batch["key_padding_mask"],
        )
        loss = causal_loss(logits, batch["labels"]) + aux * aux_coef
        clip_step(loss, optimizer, model)
        if step == 1 or step == steps or step % 20 == 0:
            print("%s step %03d  loss %.4f" % (title, step, loss.item()))


def completion_logprob(model, batch):
    """回答几个字的对数概率之和。问题位置的标签是 -100，不计入。"""
    batch = move_batch(batch, next(model.parameters()).device)
    logits, aux, _ = model(batch["input_ids"], key_padding_mask=batch["key_padding_mask"])
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = batch["labels"][:, 1:]
    picked = log_probs.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * labels.ne(-100)).sum(dim=-1), aux


def run_dpo(model, reference, pairs, steps, lr, beta, aux_coef):
    """好回答相对参考模型的提升，减去坏回答的提升。间隔越大，损失越小。"""
    optimizer = torch.optim.AdamW(trainable(model), lr=lr)
    model.train()
    reference.eval()
    for step in range(1, steps + 1):
        chosen, rejected = pairs[(step - 1) % len(pairs)]
        policy_chosen, aux_chosen = completion_logprob(model, chosen)
        policy_rejected, aux_rejected = completion_logprob(model, rejected)
        aux = aux_chosen + aux_rejected
        with torch.no_grad():
            ref_chosen, _ = completion_logprob(reference, chosen)
            ref_rejected, _ = completion_logprob(reference, rejected)
        margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
        loss = -F.logsigmoid(margin).mean() + aux * aux_coef
        clip_step(loss, optimizer, model)
        if step == 1 or step == steps or step % 20 == 0:
            print("dpo     step %03d  loss %.4f  margin %.4f" % (step, loss.item(), margin.item()))


def overlap_reward(text, target):
    """整句相同是 1。否则按目标字有多少出现在回答里，再加一点前缀重合。"""
    if text == target:
        return 1.0
    if not target:
        return 0.0
    pred = Counter(text)
    gold = Counter(target)
    matched = sum(min(pred[ch], gold[ch]) for ch in gold)
    recall = matched / len(target)
    prefix = 0
    for left, right in zip(text, target):
        if left != right:
            break
        prefix += 1
    return 0.7 * recall + 0.3 * (prefix / len(target))


def token_logprobs(model, ids, prompt_len):
    """只取回答部分每个字的对数概率。提示本身不参与策略梯度。"""
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    logits, aux, values = model(input_ids, key_padding_mask=mask)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    index = torch.arange(prompt_len - 1, len(ids) - 1, device=device)
    tokens = torch.tensor(ids[prompt_len:], dtype=torch.long, device=device)
    # 分数取回答最后一个位置，表示模型对整条回答的估计。
    return log_probs[index, tokens], aux, values[0, len(ids) - 1]


@torch.no_grad()
def sample_completion(model, tokenizer, prompt_ids, max_new, temperature):
    """按温度采样一条回答。填充和开头符的概率压掉，避免采出特殊符号。"""
    ids = list(prompt_ids)
    produced = 0
    while produced < max_new and len(ids) < model.cfg.max_seq:
        device = next(model.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        logits = model(input_ids, key_padding_mask=mask)[0][0, -1]
        logits[tokenizer.pad_id] = -1e9
        logits[tokenizer.bos_id] = -1e9
        probs = F.softmax(logits / temperature, dim=-1)
        next_id = int(torch.multinomial(probs, 1))
        ids.append(next_id)
        produced += 1
        if next_id == tokenizer.eos_id:
            break
    return ids


def run_grpo(model, reference, tokenizer, rows, sft, steps, lr, group, temperature, beta, aux_coef, value_coef):
    """采样一组回答。优势是奖励减去模型自己打的分，打分头再用均方误差去拟合奖励。"""
    optimizer = torch.optim.AdamW(trainable(model), lr=lr)
    reference.eval()
    prepared = []
    for row in rows:
        prepared.append((tokenizer.encode(row["prompt"], add_bos=True), row["prompt"], row["answer"]))

    def greedy_report():
        scores = []
        texts = []
        for _, prompt, answer in prepared:
            text = generate(model, tokenizer, prompt, max_new_tokens=24)
            texts.append(text)
            scores.append(overlap_reward(text, answer))
        model.train()
        return sum(scores) / len(scores), texts

    score, texts = greedy_report()
    print("grpo    贪心奖励 起始 %.3f  %s" % (score, " | ".join(texts)))
    for step in range(1, steps + 1):
        prompt_ids, _, answer = prepared[(step - 1) % len(prepared)]
        sampled = [
            sample_completion(model, tokenizer, prompt_ids, 16, temperature) for _ in range(group)
        ]
        rewards = torch.tensor(
            [overlap_reward(tokenizer.decode(ids[len(prompt_ids) :]), answer) for ids in sampled],
            dtype=torch.float,
            device=next(model.parameters()).device,
        )
        policy_scores = []
        kl_terms = []
        scored = []
        aux_total = torch.zeros((), device=next(model.parameters()).device)
        for ids in sampled:
            policy, aux, value = token_logprobs(model, ids, len(prompt_ids))
            aux_total = aux_total + aux
            scored.append(value)
            with torch.no_grad():
                ref, _, _ = token_logprobs(reference, ids, len(prompt_ids))
            delta = (ref - policy).clamp(-20, 20)
            policy_scores.append(policy.sum())
            # k3 估计：策略和参考模型在这条采样上的差距，两边相等时是 0。
            kl_terms.append((torch.exp(delta) - delta - 1).mean())
        policy_scores = torch.stack(policy_scores)
        values = torch.stack(scored)
        # 奖励来自和标准答案的重合。优势用模型自己的分数当基线，再在组内标准化。
        raw = rewards - values.detach()
        scale = raw.std(unbiased=False)
        advantages = torch.zeros_like(raw) if float(scale) < 1e-6 else (raw - raw.mean()) / (scale + 1e-6)
        value_loss = F.mse_loss(values, rewards)
        kl = torch.stack(kl_terms).mean()
        batch = move_batch(sft[(step - 1) % len(sft)], next(model.parameters()).device)
        sft_logits, sft_aux, _ = model(batch["input_ids"], key_padding_mask=batch["key_padding_mask"])
        sft_loss = causal_loss(sft_logits, batch["labels"])
        # 0.5 的监督损失用来拽住已经学会的改短句子，避免采样更新把它们带偏。
        loss = (
            -(advantages.detach() * policy_scores).mean()
            + beta * kl
            + 0.5 * sft_loss
            + value_coef * value_loss
            + aux_coef * (aux_total + sft_aux)
        )
        clip_step(loss, optimizer, model)
        exact = int(sum(1 for value in rewards if float(value) == 1.0))
        if step == 1 or step == steps or step % 20 == 0:
            print(
                "grpo    step %03d  loss %.4f  reward %.3f  exact %d/%d  kl %.4f"
                % (step, loss.item(), float(rewards.mean()), exact, group, float(kl))
            )
    score, texts = greedy_report()
    print("grpo    贪心奖励 结束 %.3f  %s" % (score, " | ".join(texts)))


def interleave(*groups):
    """按组轮流取一条，短的那组循环。用来在学新任务时回放旧任务。"""
    width = max(len(group) for group in groups)
    mixed = []
    for index in range(width):
        for group in groups:
            mixed.append(group[index % len(group)])
    return mixed


def save(model, tokenizer, cfg):
    """词表和权重写到同一目录。配置用字典存，加载时再还原成 Config。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUT / "tokenizer.json")
    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, OUT / "model.pt")


def main():
    """同一份参数依次学会接字、按指令答、偏向更好的采样、看图，以及自己决定要不要调工具。"""
    if run_tool("加:2:3") != "5" or run_tool("减:7:4") != "3" or run_tool("乘:2:3") != "6":
        raise RuntimeError("工具计算不正确")

    cfg = Config()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    pick_device()
    tokenizer = BPETokenizer.fit(all_texts(), cfg.bpe_merges)
    model = TinyLM(cfg, tokenizer.vocab_size)
    model = place(model)
    total = sum(param.numel() for param in model.parameters())
    moe = model.blocks[cfg.moe_layer].mlp
    if not isinstance(moe, MoE):
        raise RuntimeError("moe_layer 指向的层不是专家层")
    expert_params = sum(param.numel() for param in moe.experts[0].parameters())
    active = total - (cfg.n_experts - cfg.top_k) * expert_params
    if total >= 1_000_000:
        human = "%.2f 亿" % (total / 1e8)
    else:
        human = "%.2f 万" % (total / 1e4)
    print(
        "参数量 %d，合 %.6fB（%s）  单次前向大约用到 %d  词表 %d  设备 %s"
        % (total, total / 1e9, human, active, tokenizer.vocab_size, next(model.parameters()).device)
    )

    corpus = hf_text_batches(tokenizer, cfg.max_seq)
    if corpus:
        run_lm(model, corpus, cfg.hf_text_steps, cfg.lr, "hf-text", cfg.moe_aux)
    text = sft_batches(tokenizer)
    run_lm(model, pretrain_batches(tokenizer), cfg.pretrain_steps, cfg.lr, "pretrain", cfg.moe_aux)
    run_lm(model, text, cfg.sft_steps, cfg.lr, "sft    ", cfg.moe_aux)

    if cfg.grpo_steps > 0:
        grpo_reference = copy.deepcopy(model)
        for param in grpo_reference.parameters():
            param.requires_grad_(False)
        run_grpo(
            model,
            grpo_reference,
            tokenizer,
            grpo_rows(),
            text,
            cfg.grpo_steps,
            cfg.lr,
            cfg.grpo_group,
            cfg.grpo_temperature,
            cfg.grpo_beta,
            cfg.moe_aux,
            cfg.value_coef,
        )

    reference = copy.deepcopy(model)
    for param in reference.parameters():
        param.requires_grad_(False)
    run_dpo(model, reference, dpo_pairs(tokenizer), cfg.dpo_steps, cfg.lr, cfg.dpo_beta, cfg.moe_aux)

    apply_lora(model, cfg.lora_rank, cfg.lora_alpha)
    lora_params = sum(param.numel() for param in trainable(model))
    print("lora    可训练参数 %d" % lora_params)
    adapted = lora_batches(tokenizer)
    run_lm(model, adapted, cfg.lora_steps, cfg.lr, "lora   ", cfg.moe_aux)
    merge_lora(model)

    # 看图是新能力，梯度大。先学会颜色，再学工具调用，回放时颜色的损失已经很小。
    colors = mm_batches(tokenizer, cfg.image_size, cfg.patch_size)
    photos = hf_image_batches(tokenizer, cfg.image_size, cfg.patch_size, cfg.max_seq)
    if photos:
        run_lm(model, photos, cfg.hf_image_steps, cfg.lr, "hf-img ", cfg.moe_aux)
    run_lm(model, interleave(colors, text, adapted), cfg.mm_steps, cfg.lr, "mm     ", cfg.moe_aux)
    clips = hf_video_batches(tokenizer, cfg.image_size, cfg.patch_size, cfg.max_seq)
    if clips:
        run_lm(model, clips, cfg.hf_video_steps, cfg.lr, "hf-vid ", cfg.moe_aux)

    tool = tool_batches(tokenizer)
    run_lm(model, tool, cfg.tool_only_steps, cfg.lr, "tool   ", cfg.moe_aux)

    # 每一步同时更新改短、颜色、物体和算式，并留下对答案最多的那份权重。
    groups = [text, colors, tool]
    if photos:
        groups.append(photos)
    best_score = -1
    best_state = None

    def consider(tag):
        nonlocal best_score, best_state
        score, brief = probe(model, tokenizer, cfg)
        print("检查 %s  对上 %d  %s" % (tag, score, brief))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.train()

    consider("分开学完")
    joint_steps = cfg.polish_steps
    optimizer = torch.optim.AdamW(trainable(model), lr=cfg.lr)
    device = next(model.parameters()).device
    groups = [fuse_batches(group, device) for group in groups]
    for group in groups:
        shape = group[0]["input_ids"].shape
        print("joint    batch %d x %d" % (shape[0], shape[1]))
    model.train()
    for step in range(1, joint_steps + 1):
        optimizer.zero_grad()
        total = 0.0
        for group in groups:
            batch = group[(step - 1) % len(group)]
            logits, aux, _ = model(
                batch["input_ids"],
                images=batch.get("images"),
                key_padding_mask=batch["key_padding_mask"],
            )
            loss = causal_loss(logits, batch["labels"]) + aux * cfg.moe_aux
            (loss / len(groups)).backward()
            total += float(loss.detach())
        apply_grads(optimizer, model)
        if step == 1 or step == joint_steps or step % 20 == 0:
            print("joint   step %03d  loss %.4f" % (step, total / len(groups)))
            consider("joint %d" % step)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("用对上最多的权重，共 %d 项" % best_score)
    save(model, tokenizer, cfg)
    print("已保存到 %s" % OUT)

    for row in read_jsonl(DATA / "sft.jsonl"):
        print("改短:", act(model, tokenizer, row["prompt"]))
    for row in read_jsonl(DATA / "lora.jsonl"):
        print("适配:", act(model, tokenizer, row["prompt"]))
    for row in read_jsonl(DATA / "dialogue.jsonl"):
        print("对话:", act(model, tokenizer, row["prompt"]))
    question = "这张图是什么颜色？\n回答："
    for color in ("红", "绿", "蓝"):
        image = solid_image(color, cfg.image_size)
        print("看图 %s:" % color, act(model, tokenizer, question, image=image))
    sided = split_image("红", "蓝", cfg.image_size)
    print("左边:", act(model, tokenizer, "这张图左边是什么颜色？\n回答：", image=sided))
    print("右边:", act(model, tokenizer, "这张图右边是什么颜色？\n回答：", image=sided))
    seen = set()
    image_path = HF / "images.jsonl"
    if image_path.exists():
        for row in read_jsonl(image_path):
            if row["answer"] in seen:
                continue
            seen.add(row["answer"])
            photo = _load_rgb(HF / row["file"], cfg.image_size)
            print("图片 %s:" % row["answer"], act(model, tokenizer, row["prompt"], image=photo))
            if len(seen) >= 5:
                break
    print("工具 2加3:", act(model, tokenizer, "算式2加3。\n回答："))
    print("工具 2乘3:", act(model, tokenizer, "算式2乘3。\n回答："))
    print("工具 2加3减1:", act(model, tokenizer, "算式2加3减1。\n回答："))
    print("未见 4加2:", act(model, tokenizer, "算式4加2。\n回答："))
    weather = "指令：把下面这句话改短。\n内容：今天天气非常好，很适合出门散步。\n回答："
    print("采样:", act(model, tokenizer, weather, temperature=0.8))
    reason()


def show(model, tokenizer, cfg):
    """把改短、颜色、物体和算式各打几条，方便对照标准答案。"""
    for row in read_jsonl(DATA / "sft.jsonl"):
        print("改短:", act(model, tokenizer, row["prompt"]))
    for row in read_jsonl(DATA / "dialogue.jsonl"):
        print("对话:", act(model, tokenizer, row["prompt"]))
    question = "这张图是什么颜色？\n回答："
    for color in ("红", "绿", "蓝"):
        image = solid_image(color, cfg.image_size)
        print("看图 %s:" % color, act(model, tokenizer, question, image=image))
    sided = split_image("红", "蓝", cfg.image_size)
    print("左边:", act(model, tokenizer, "这张图左边是什么颜色？\n回答：", image=sided))
    print("右边:", act(model, tokenizer, "这张图右边是什么颜色？\n回答：", image=sided))
    seen = set()
    for row in read_jsonl(HF / "images.jsonl"):
        if row["answer"] in seen:
            continue
        seen.add(row["answer"])
        photo = _load_rgb(HF / row["file"], cfg.image_size)
        print("图片 %s:" % row["answer"], act(model, tokenizer, row["prompt"], image=photo))
        if len(seen) >= 5:
            break
    print("工具 2加3:", act(model, tokenizer, "算式2加3。\n回答："))
    print("工具 2乘3:", act(model, tokenizer, "算式2乘3。\n回答："))


def tool_gold(row):
    """程序接上每一次结果后，模型应写出的整段，含最后读出的数字。"""
    steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
    text = ""
    for step in steps:
        text += step["call"] + "<结果>" + step["result"] + "</结果>"
    return text + row["answer"]


def seq_probe(model, tokenizer, cfg):
    """文本、全部算式和看图一起计分。连算必须把第二次调用也写出来。"""
    checks = []
    for row in read_jsonl(DATA / "sft.jsonl")[:4]:
        checks.append((row["answer"], act(model, tokenizer, row["prompt"])))
    for row in read_jsonl(DATA / "tool.jsonl"):
        checks.append((tool_gold(row), act(model, tokenizer, row["prompt"])))
    question = "这张图是什么颜色？\n回答："
    for color, answer in (("红", "红色"), ("绿", "绿色"), ("蓝", "蓝色")):
        image = solid_image(color, cfg.image_size)
        checks.append((answer, act(model, tokenizer, question, image=image)))
    sided = split_image("红", "蓝", cfg.image_size)
    checks.append(("红色", act(model, tokenizer, "这张图左边是什么颜色？\n回答：", image=sided)))
    checks.append(("蓝色", act(model, tokenizer, "这张图右边是什么颜色？\n回答：", image=sided)))
    seen = set()
    for row in read_jsonl(HF / "images.jsonl"):
        if row["answer"] in seen:
            continue
        seen.add(row["answer"])
        photo = _load_rgb(HF / row["file"], cfg.image_size)
        checks.append((row["answer"], act(model, tokenizer, row["prompt"], image=photo)))
        if len(seen) >= 5:
            break
    score = sum(1 for gold, pred in checks if pred == gold)
    # 摘要只留两道连算，方便看有没有写出第二步。
    chain = [pred.replace("\n", "")[:36] for _, pred in checks[15:17]]
    return score, " ".join(chain)


def show_tasks(model, tokenizer):
    """把改短、对话、每种运算和两道连算逐条打出来。"""
    print("== 文本 ==")
    for name in ("sft.jsonl", "dialogue.jsonl", "lora.jsonl"):
        for row in read_jsonl(DATA / name):
            pred = act(model, tokenizer, row["prompt"])
            mark = "对" if pred == row["answer"] else "错"
            print("%s %s => %s" % (mark, row["answer"], pred))
    print("== 算子 ==")
    for row in read_jsonl(DATA / "tool.jsonl"):
        pred = act(model, tokenizer, row["prompt"])
        gold = tool_gold(row)
        mark = "对" if pred == gold else "错"
        title = row["prompt"].split("。")[0]
        print("%s %s => %s" % (mark, title, pred))
    print("== 未见 ==")
    for prompt in ("算式4加2。\n回答：", "算式6减3。\n回答：", "算式5乘2。\n回答："):
        print("%s => %s" % (prompt.split("。")[0], act(model, tokenizer, prompt)))


def set_trainable(model, prefixes):
    """只打开名字以这些前缀开头的参数。前面的层保持不动，避免把已经对上的句子冲掉。"""
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)


def snapshot(model):
    """整份权重拷到 CPU，后面分数变差时再换回来。"""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def refine_phase(model, tokenizer, cfg, slots, prefixes, steps, lr, title, best_score, best_state, probe_fn=None):
    """一小段更新。每一步都对答案，只有条数变多才留下。"""
    set_trainable(model, prefixes)
    optimizer = torch.optim.AdamW(trainable(model), lr=lr)
    device = next(model.parameters()).device
    weight_sum = sum(weight for _, weight in slots)
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        total = 0.0
        model.train()
        for batch, weight in slots:
            batch = move_batch(batch, device)
            logits, aux, _ = model(
                batch["input_ids"],
                images=batch.get("images"),
                key_padding_mask=batch["key_padding_mask"],
            )
            loss = causal_loss(logits, batch["labels"]) + aux * cfg.moe_aux
            (loss * weight / weight_sum).backward()
            total += float(loss.detach()) * weight
        apply_grads(optimizer, model)
        checker = probe if probe_fn is None else probe_fn
        score, brief = checker(model, tokenizer, cfg)
        print("%s step %03d  loss %.4f  对上 %d  %s" % (title, step, total / weight_sum, score, brief))
        if score > best_score:
            best_score = score
            best_state = snapshot(model)
    model.load_state_dict(best_state)
    for param in model.parameters():
        param.requires_grad = True
    return best_score, best_state


def refine():
    """从当前权重接着训。先只改最后几层，让它照着结果数字说；再让不同物体在同一批里分开。"""
    from generate import load

    model, tokenizer, cfg = load()
    model = place(model)
    tail = ["blocks.%d." % index for index in range(cfg.n_layers - 4, cfg.n_layers)] + ["norm."]
    ops = tool_op_batches(tokenizer)
    colors = mm_batches(tokenizer, cfg.image_size, cfg.patch_size)
    text = sft_batches(tokenizer)
    best_score, _ = probe(model, tokenizer, cfg)
    best_state = snapshot(model)
    print("接着训，当前对上 %d" % best_score)
    # 词嵌入不动，最后四层学习：看见 <结果>6</结果> 就说 6，同时别把调用写成别的算式。
    best_score, best_state = refine_phase(
        model,
        tokenizer,
        cfg,
        [
            (tool_echo_contrast(tokenizer), 1.0),
            (ops["加"][0], 0.5),
            (ops["乘"][0], 0.5),
        ],
        tail,
        16,
        5e-5,
        "读数字",
        best_score,
        best_state,
    )
    photos = photo_contrast_batch(tokenizer, cfg.image_size, cfg.patch_size, cfg.max_seq)
    # 图像块和最后四层一起学。十个类别在同一批里，损失会把「鸟」从其他类上拉开。
    best_score, best_state = refine_phase(
        model,
        tokenizer,
        cfg,
        [
            (photos, 1.0),
            (text[0], 0.3),
            (colors[0], 0.3),
        ],
        ["vision."] + tail,
        16,
        3e-5,
        "分物体",
        best_score,
        best_state,
    )
    model.load_state_dict(best_state)
    print("用对上最多的权重，共 %d 项" % best_score)
    save(model, tokenizer, cfg)
    show(model, tokenizer, cfg)


def focus():
    """接着当前权重，只把飞机、鸟和它们被说成的类别放在同一批里分开。"""
    from generate import load

    model, tokenizer, cfg = load()
    model = place(model)
    tail = ["blocks.%d." % index for index in range(cfg.n_layers - 4, cfg.n_layers)] + ["norm."]
    photos = photo_focus_batch(
        tokenizer,
        cfg.image_size,
        cfg.patch_size,
        cfg.max_seq,
        ["飞机", "汽车", "鸟", "青蛙", "马"],
        copies=2,
    )
    text = sft_batches(tokenizer)
    colors = mm_batches(tokenizer, cfg.image_size, cfg.patch_size)
    best_score, _ = probe(model, tokenizer, cfg)
    best_state = snapshot(model)
    print("接着分开物体，当前对上 %d" % best_score)
    best_score, best_state = refine_phase(
        model,
        tokenizer,
        cfg,
        [
            (photos, 1.0),
            (text[0], 0.3),
            (colors[0], 0.3),
        ],
        ["vision."] + tail,
        12,
        2e-5,
        "拉开",
        best_score,
        best_state,
    )
    model.load_state_dict(best_state)
    print("用对上最多的权重，共 %d 项" % best_score)
    save(model, tokenizer, cfg)
    show(model, tokenizer, cfg)


def sequences():
    """接着当前权重学连算的第二步，并让加减乘在同一步里对着记住调用和结果数字。"""
    from generate import load

    model, tokenizer, cfg = load()
    model = place(model)
    tail = ["blocks.%d." % index for index in range(cfg.n_layers - 4, cfg.n_layers)] + ["norm."]
    best_score, _ = seq_probe(model, tokenizer, cfg)
    best_state = snapshot(model)
    print("接着训序列，当前对上 %d" % best_score)
    best_score, best_state = refine_phase(
        model,
        tokenizer,
        cfg,
        [
            (tool_chain_batch(tokenizer), 1.0),
            (text_hold_batch(tokenizer), 0.8),
        ],
        tail,
        12,
        1.5e-5,
        "序列",
        best_score,
        best_state,
        probe_fn=seq_probe,
    )
    model.load_state_dict(best_state)
    print("用对上最多的权重，共 %d 项" % best_score)
    save(model, tokenizer, cfg)
    show_tasks(model, tokenizer)


def grow_embeddings(model, vocab_size):
    """词表变长时把新字的向量接在后面。旧字的向量原样保留，输出层仍和词嵌入绑在一起。"""
    old = model.tok_emb
    if old.num_embeddings == vocab_size:
        return
    device = old.weight.device
    fresh = torch.nn.Embedding(vocab_size, old.embedding_dim)
    fresh.weight.data.normal_(mean=0.0, std=0.02)
    fresh.weight.data[: old.num_embeddings] = old.weight.data
    model.tok_emb = fresh.to(device)
    model.lm_head.weight = model.tok_emb.weight


def reason_rows(specs):
    """把若干道算式收成（问题，完整回答）。完整回答含思考、调用、程序填的结果和复读。"""
    rows = []
    for op, left, right in specs:
        prompt, _call, _echo, _result, full = reason_example(op, left, right)
        rows.append((prompt, full))
    return rows


def reason_score(model, tokenizer, rows):
    """整段对上，以及思考和调用这两段是否先写对。摘要只留前两条。"""
    full_ok = 0
    call_ok = 0
    shown = []
    for prompt, full in rows:
        pred = act(model, tokenizer, prompt)
        call = full.split("<结果>")[0]
        full_ok += int(pred == full)
        call_ok += int(pred.startswith(call))
        if len(shown) < 2:
            shown.append(pred.replace("\n", "")[:28])
    return full_ok, call_ok, " ".join(shown)


def reason():
    """学会先写出取数，再调用工具，最后照程序给出的结果说。留出的格子用来检查能不能套到新算式。"""
    from generate import load

    model, tokenizer, cfg = load()
    tokenizer.add_chars("思考先取")
    grow_embeddings(model, tokenizer.vocab_size)
    model = place(model)
    for param in model.parameters():
        param.requires_grad = True
    # 上一轮只开最后几层，还冻住了旧字的输出向量，损失停在 3 附近，生成仍是背下来的「乘:6:6」。
    # 思考格式大部分是原来的字，整网一起改才能把「先取」写到工具前面。
    calls, echos, holdouts = reason_batches(tokenizer, chunk=32)
    seen = reason_rows((("乘", 2, 3), ("乘", 4, 2), ("加", 2, 3), ("减", 7, 4), ("乘", 9, 8), ("加", 6, 1)))
    # 留出的格子不参与更新。分数先看这些新算式，再看训练里见过的几道。
    best_hold, best_hold_call, _ = reason_score(model, tokenizer, holdouts)
    best_full, best_call, _ = reason_score(model, tokenizer, seen)
    best_state = snapshot(model)
    print(
        "接着学推理，留出整段 %d/%d，见过整段 %d/%d"
        % (best_hold, len(holdouts), best_full, len(seen))
    )
    # 3e-4 会把新字「思」写成重复。这里用小学习率，并且多看几遍「结果后面的数字」。
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-4)
    device = next(model.parameters()).device
    calls = fuse_batches(calls, device)
    echos = fuse_batches(echos, device)
    call_shape = calls[0]["input_ids"].shape
    echo_shape = echos[0]["input_ids"].shape
    print("推理 batch 调用 %d x %d，复读 %d x %d" % (call_shape[0], call_shape[1], echo_shape[0], echo_shape[1]))
    steps = 300
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        total = 0.0
        model.train()
        slots = [
            (calls[(step - 1) % len(calls)], 1.0),
            (echos[(step - 1) % len(echos)], 2.0),
        ]
        weight_sum = sum(weight for _, weight in slots)
        for batch, weight in slots:
            logits, aux, _ = model(
                batch["input_ids"],
                images=batch.get("images"),
                key_padding_mask=batch["key_padding_mask"],
            )
            loss = causal_loss(logits, batch["labels"]) + aux * cfg.moe_aux
            (loss * weight / weight_sum).backward()
            total += float(loss.detach()) * weight
        apply_grads(optimizer, model)
        if step == 1 or step == steps or step % 100 == 0:
            hold_full, hold_call, hold_brief = reason_score(model, tokenizer, holdouts)
            full_ok, call_ok, brief = reason_score(model, tokenizer, seen)
            print(
                "推理 step %03d  loss %.4f  留出 %d/%d  见过 %d/%d  %s"
                % (step, total / weight_sum, hold_full, len(holdouts), full_ok, len(seen), brief or hold_brief)
            )
            if (hold_full, full_ok, hold_call) > (best_hold, best_full, best_hold_call):
                best_hold, best_full, best_hold_call = hold_full, full_ok, hold_call
                best_state = snapshot(model)
    model.load_state_dict(best_state)
    for param in model.parameters():
        param.requires_grad = True
    print("用对上最多的权重，留出整段 %d/%d，见过整段 %d/%d" % (best_hold, len(holdouts), best_full, len(seen)))
    save(model, tokenizer, cfg)
    print("== 留出 ==")
    for prompt, full in holdouts:
        pred = act(model, tokenizer, prompt)
        mark = "对" if pred == full else "错"
        print("%s %s => %s" % (mark, prompt.split("。")[0], pred))
    print("== 九九 ==")
    exact = 0
    for left in range(1, 10):
        row = []
        for right in range(1, 10):
            prompt = "算式%d乘%d。\n回答：" % (left, right)
            pred = act(model, tokenizer, prompt)
            product = left * right
            gold = "<思考>先取%d乘%d</思考><工具>乘:%d:%d</工具><结果>%d</结果>%d" % (
                left,
                right,
                left,
                right,
                product,
                product,
            )
            ok = pred == gold
            exact += int(ok)
            row.append("%d" % product if ok else pred.split("</结果>")[-1][:4])
        print("%d | %s" % (left, " ".join(row)))
    print("九九整段对上 %d/81" % exact)
    for row in read_jsonl(DATA / "sft.jsonl")[:4]:
        pred = act(model, tokenizer, row["prompt"])
        mark = "对" if pred == row["answer"] else "错"
        print("%s 文本 %s => %s" % (mark, row["answer"], pred))


def _video_rows(image_size):
    path = HF / "videos.jsonl"
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        frames = torch.stack([_load_rgb(HF / name, image_size) for name in row["frames"]])
        rows.append((row["prompt"], row["answer"], frames))
    return rows


def english_checks(tokenizer, limit=8):
    """留几句英文，看改短后是不是更短、而且字还在原句里。"""
    checks = []
    path = HF / "text.txt"
    if not path.exists():
        return checks
    for line in path.read_text(encoding="utf-8").splitlines():
        for piece in line.replace("!", ".").replace("?", ".").split(". "):
            content = " ".join(piece.strip(" .").split())
            if len(content) < 40 or len(content) > 90:
                continue
            answer = _compress(content)
            if answer is None:
                continue
            prompt = "指令：把下面这句话改短。\n内容：%s\n回答：" % content
            try:
                if len(tokenizer.encode(prompt)) > 64:
                    continue
            except KeyError:
                continue
            checks.append((prompt, content))
            if len(checks) >= limit:
                return checks
    return checks


def _short_ok(content, pred):
    pred = pred.strip()
    if len(pred) < 2 or len(pred) >= len(content):
        return False
    if "指令" in pred or "回答" in pred:
        return False
    kept = sum(1 for ch in pred if ch in content or ch.isspace())
    return kept / len(pred) >= 0.8


def task_score(model, tokenizer, op_rows, videos, english):
    """改短原题、英文抽查、加减乘整段、视频类别，各算对上的条数。"""
    text_ok = 0
    for row in read_jsonl(DATA / "sft.jsonl"):
        pred = act(model, tokenizer, row["prompt"])
        text_ok += int(pred == row["answer"])
    eng_ok = 0
    for prompt, content in english:
        pred = act(model, tokenizer, prompt)
        eng_ok += int(_short_ok(content, pred))
    op_ok, _, brief = reason_score(model, tokenizer, op_rows)
    vid_ok = 0
    for prompt, answer, frames in videos:
        pred = act(model, tokenizer, prompt, image=frames)
        vid_ok += int(pred == answer)
    return text_ok, eng_ok, op_ok, vid_ok, brief


def tasks():
    """接着当前权重学三件事：把句子改短、分清加减乘、认出视频里的动作。"""
    from generate import load

    model, tokenizer, cfg = load()
    model = place(model)
    for param in model.parameters():
        param.requires_grad = True
    text = shorten_batches(tokenizer, cfg.max_seq)
    calls, echos, _ = reason_batches(tokenizer, chunk=32, keep_holdout=True)
    videos_raw = hf_video_batches(tokenizer, cfg.image_size, cfg.patch_size, cfg.max_seq)
    op_rows = reason_rows(
        (
            ("加", 2, 3),
            ("加", 7, 8),
            ("加", 9, 1),
            ("减", 7, 4),
            ("减", 9, 2),
            ("减", 8, 5),
            ("乘", 2, 3),
            ("乘", 7, 8),
            ("乘", 4, 9),
        )
    )
    videos = _video_rows(cfg.image_size)
    english = english_checks(tokenizer)
    text_ok, eng_ok, op_ok, vid_ok, _ = task_score(model, tokenizer, op_rows, videos, english)
    best = text_ok + eng_ok + op_ok + vid_ok
    best_state = snapshot(model)
    print("接着学文本/算子/视频，当前 中文 %d  英文 %d  算子 %d  视频 %d" % (text_ok, eng_ok, op_ok, vid_ok))
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-4)
    device = next(model.parameters()).device
    text_batches = fuse_batches(text, device, target=32)
    groups = [
        text_batches,
        text_batches,
        fuse_batches(calls, device),
        fuse_batches(echos, device),
    ]
    if videos_raw:
        groups.append(fuse_batches(videos_raw, device, target=48))
    for index, group in enumerate(groups):
        shape = group[0]["input_ids"].shape
        print("任务 batch %d  %d x %d" % (index, shape[0], shape[1]))
    steps = 200
    cursors = [0] * len(groups)
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        total = 0.0
        model.train()
        weight_sum = float(len(groups))
        for index, group in enumerate(groups):
            batch = group[cursors[index] % len(group)]
            cursors[index] += 1
            logits, aux, _ = model(
                batch["input_ids"],
                images=batch.get("images"),
                key_padding_mask=batch["key_padding_mask"],
            )
            loss = causal_loss(logits, batch["labels"]) + aux * cfg.moe_aux
            (loss / weight_sum).backward()
            total += float(loss.detach())
        apply_grads(optimizer, model)
        if step == 1 or step == steps or step % 100 == 0:
            text_ok, eng_ok, op_ok, vid_ok, brief = task_score(model, tokenizer, op_rows, videos, english)
            print(
                "任务 step %03d  loss %.4f  中文 %d/8  英文 %d/%d  算子 %d/%d  视频 %d/%d  %s"
                % (
                    step,
                    total / weight_sum,
                    text_ok,
                    eng_ok,
                    len(english),
                    op_ok,
                    len(op_rows),
                    vid_ok,
                    len(videos),
                    brief,
                )
            )
            score = text_ok + eng_ok + op_ok + vid_ok
            if score > best:
                best = score
                best_state = snapshot(model)
    model.load_state_dict(best_state)
    for param in model.parameters():
        param.requires_grad = True
    text_ok, eng_ok, op_ok, vid_ok, _ = task_score(model, tokenizer, op_rows, videos, english)
    print(
        "用对上最多的权重，中文 %d/8  英文 %d/%d  算子 %d/%d  视频 %d/%d"
        % (text_ok, eng_ok, len(english), op_ok, len(op_rows), vid_ok, len(videos))
    )
    save(model, tokenizer, cfg)
    print("== 文本 ==")
    for row in read_jsonl(DATA / "sft.jsonl"):
        pred = act(model, tokenizer, row["prompt"])
        mark = "对" if pred == row["answer"] else "错"
        print("%s %s => %s" % (mark, row["answer"], pred))
    print("== 英文 ==")
    for prompt, content in english:
        pred = act(model, tokenizer, prompt)
        mark = "对" if _short_ok(content, pred) else "错"
        print("%s %s => %s" % (mark, content[:48], pred.replace("\n", "")[:48]))
    print("== 算子 ==")
    for op in ("加", "减", "乘"):
        exact = 0
        for left in range(1, 10):
            for right in range(1, 10):
                prompt, _call, _echo, _result, full = reason_example(op, left, right)
                exact += int(act(model, tokenizer, prompt) == full)
        print("%s %d/81" % (op, exact))
    print("== 视频 ==")
    for prompt, answer, frames in videos:
        pred = act(model, tokenizer, prompt, image=frames)
        mark = "对" if pred == answer else "错"
        print("%s %s => %s" % (mark, answer, pred))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "reason":
        reason()
    elif command == "tasks":
        tasks()
    else:
        main()
