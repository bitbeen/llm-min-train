"""把 samples/ 里的句子、偏好、图片和工具调用收成一个个批次。"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vision import make_image

# 样本和这份代码放在一起。
DATA = Path(__file__).resolve().parent / "samples"
# Hugging Face 下载后落在这里，和手写样本分开。
HF = Path(__file__).resolve().parent / "data" / "hf"


def read_jsonl(path):
    """一行一个 JSON 对象。空行跳过。"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _strings(value):
    """把嵌套的字段摊成字符串，词表才能收进工具的多步调用。"""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def all_texts():
    """词表只从这些文件收集。新出现的字要先写进来，否则编码会报错。"""
    texts = [(DATA / "pretrain.txt").read_text(encoding="utf-8")]
    for name in ("sft.jsonl", "dpo.jsonl", "mm.jsonl", "grpo.jsonl", "lora.jsonl", "tool.jsonl", "dialogue.jsonl"):
        for row in read_jsonl(DATA / name):
            texts.extend(_strings(row))
    text_path = HF / "text.txt"
    if text_path.exists():
        texts.append(text_path.read_text(encoding="utf-8"))
    for name in ("images.jsonl", "videos.jsonl"):
        path = HF / name
        if path.exists():
            for row in read_jsonl(path):
                texts.extend(_strings(row))
    return texts


def _pad(rows, pad_id):
    """把不等长的样本补到同一长度。补出来的位置标签保持 -100，不参与损失。"""
    width = max(len(row[0]) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, (ids, labs) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(labs)] = torch.tensor(labs)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "key_padding_mask": input_ids.eq(pad_id),
    }


def pretrain_batches(tokenizer):
    """一整句既是输入也是标签，用来学下一个字。"""
    batches = []
    for line in (DATA / "pretrain.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids = tokenizer.encode(line, add_bos=True, add_eos=True)
        batches.append(_pad([(ids, ids)], tokenizer.pad_id))
    return batches


def _prompt_answer(tokenizer, prompt, answer):
    """问题部分的标签是 -100，损失只落在回答和结尾符上。"""
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(answer, add_eos=True)
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return ids, labels


def sft_batches(tokenizer):
    """指令回答，外加多轮对话。多轮样本的损失只落在最后一轮回答上。"""
    batches = []
    for name in ("sft.jsonl", "dialogue.jsonl"):
        for row in read_jsonl(DATA / name):
            batches.append(_pad([_prompt_answer(tokenizer, row["prompt"], row["answer"])], tokenizer.pad_id))
    return batches


def dpo_pairs(tokenizer):
    """同一问题的好回答和坏回答配成一对。"""
    pairs = []
    for row in read_jsonl(DATA / "dpo.jsonl"):
        chosen = _pad([_prompt_answer(tokenizer, row["prompt"], row["chosen"])], tokenizer.pad_id)
        rejected = _pad([_prompt_answer(tokenizer, row["prompt"], row["rejected"])], tokenizer.pad_id)
        pairs.append((chosen, rejected))
    return pairs


def grpo_rows():
    """采样时用的问题和标准答案。奖励按回答和这条答案的重合来打。"""
    return read_jsonl(DATA / "grpo.jsonl")


def lora_batches(tokenizer):
    """LoRA 阶段要新记住的改短句子，格式和监督微调一样。"""
    batches = []
    for row in read_jsonl(DATA / "lora.jsonl"):
        batches.append(_pad([_prompt_answer(tokenizer, row["prompt"], row["answer"])], tokenizer.pad_id))
    return batches


def _tool_trace(tokenizer, prompt, steps, answer):
    """一步调用一条样本，最后一条才是最终数字。调用重复一次，避免模型跳过算式。"""
    batches = []
    prefix = prompt
    for step in steps:
        call = _pad([_prompt_answer(tokenizer, prefix, step["call"])], tokenizer.pad_id)
        batches.append(call)
        batches.append(call)
        prefix = prefix + step["call"] + "<结果>" + step["result"] + "</结果>"
    batches.append(_pad([_prompt_answer(tokenizer, prefix, answer)], tokenizer.pad_id))
    return batches


def tool_op_batches(tokenizer):
    """把第一次调用按加、减、乘分开。三种运算要在同一步里对着学。"""
    groups = {"加": [], "减": [], "乘": []}
    for row in read_jsonl(DATA / "tool.jsonl"):
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        call = steps[0]["call"]
        op = call[len("<工具>")]
        groups[op].append(_pad([_prompt_answer(tokenizer, row["prompt"], call)], tokenizer.pad_id))
    return groups


def tool_echo_batches(tokenizer):
    """结果已经接在上下文里之后，回答必须是这个数字，不能套用另一道题。"""
    groups = {}
    for row in read_jsonl(DATA / "tool.jsonl"):
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        prefix = row["prompt"]
        for step in steps:
            prefix = prefix + step["call"] + "<结果>" + step["result"] + "</结果>"
        answer = row["answer"]
        groups.setdefault(answer, []).append(
            _pad([_prompt_answer(tokenizer, prefix, answer)], tokenizer.pad_id)
        )
    return groups


def tool_call_batch(tokenizer, step_index):
    """第 step_index 次调用放进同一批。0 是看到算式就写的第一次，1 是结果接上后的下一步。"""
    rows = []
    for row in read_jsonl(DATA / "tool.jsonl"):
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        if len(steps) <= step_index:
            continue
        prefix = row["prompt"]
        for step in steps[:step_index]:
            prefix = prefix + step["call"] + "<结果>" + step["result"] + "</结果>"
        # 连算的第二步只有两条，重复一次，避免被十几个第一次调用盖住。
        times = 2 if step_index > 0 else 1
        for _ in range(times):
            rows.append(_prompt_answer(tokenizer, prefix, steps[step_index]["call"]))
    return _pad(rows, tokenizer.pad_id)


def tool_all_echo(tokenizer):
    """每道题在完整结果后面都要说出自己的数字，包括减法和连算的最后一步。"""
    rows = []
    for row in read_jsonl(DATA / "tool.jsonl"):
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        prefix = row["prompt"]
        for step in steps:
            prefix = prefix + step["call"] + "<结果>" + step["result"] + "</结果>"
        rows.append(_prompt_answer(tokenizer, prefix, row["answer"]))
    return _pad(rows, tokenizer.pad_id)


def tool_branch_batch(tokenizer):
    """结果接上之后放进同一批。单步题的标签是数字，连算题的标签是下一次调用。"""
    rows = []
    for row in read_jsonl(DATA / "tool.jsonl"):
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        prefix = row["prompt"]
        for index, step in enumerate(steps):
            prefix = prefix + step["call"] + "<结果>" + step["result"] + "</结果>"
            if index + 1 < len(steps):
                rows.append(_prompt_answer(tokenizer, prefix, steps[index + 1]["call"]))
            else:
                rows.append(_prompt_answer(tokenizer, prefix, row["answer"]))
    return _pad(rows, tokenizer.pad_id)


def tool_fix_calls(tokenizer):
    """调用写错的两道，配上已经写对的加、减、乘，避免算子和数字被改成同一个。"""
    want = {
        "算式8减5。\n回答：",
        "算式5加4。\n回答：",
        "算式2加3。\n回答：",
        "算式7减4。\n回答：",
        "算式2乘3。\n回答：",
    }
    rows = []
    for row in read_jsonl(DATA / "tool.jsonl"):
        if row["prompt"] not in want:
            continue
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        rows.append(_prompt_answer(tokenizer, row["prompt"], steps[0]["call"]))
    return _pad(rows, tokenizer.pad_id)


def tool_chain_batch(tokenizer):
    """连算的第二步重复多遍，已经对上的单步题各留一条，避免结果后面一律改成调用或一律报数字。"""
    anchors = {
        "算式2加3。\n回答：",
        "算式9加1。\n回答：",
        "算式4加6。\n回答：",
        "算式3加2。\n回答：",
        "算式2乘3。\n回答：",
        "算式3乘2。\n回答：",
    }
    rows = []
    for row in read_jsonl(DATA / "tool.jsonl"):
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        if len(steps) == 1:
            if row["prompt"] not in anchors:
                continue
            prefix = row["prompt"] + steps[0]["call"] + "<结果>" + steps[0]["result"] + "</结果>"
            rows.append(_prompt_answer(tokenizer, prefix, row["answer"]))
            continue
        prefix = row["prompt"] + steps[0]["call"] + "<结果>" + steps[0]["result"] + "</结果>"
        for _ in range(8):
            rows.append(_prompt_answer(tokenizer, prefix, steps[1]["call"]))
        done = prefix + steps[1]["call"] + "<结果>" + steps[1]["result"] + "</结果>"
        rows.append(_prompt_answer(tokenizer, done, row["answer"]))
    return _pad(rows, tokenizer.pad_id)


# 这些格子不进训练，用来看模型是不是按算式取数，而不是背答案。
REASON_HOLDOUT = {
    ("乘", 7, 8),
    ("乘", 9, 9),
    ("乘", 6, 7),
    ("乘", 4, 9),
    ("乘", 8, 3),
    ("乘", 5, 5),
    ("乘", 1, 1),
    ("乘", 3, 6),
    ("加", 7, 8),
    ("减", 9, 2),
}


def _reason_ops():
    return {"加": lambda a, b: a + b, "减": lambda a, b: a - b, "乘": lambda a, b: a * b}


def reason_example(op, left, right):
    """先把两个数和运算符抄出来，再写调用。结果由程序填，回答只复读这个结果。"""
    prompt = "算式%d%s%d。\n回答：" % (left, op, right)
    call = "<思考>先取%d%s%d</思考><工具>%s:%d:%d</工具>" % (left, op, right, op, left, right)
    result = str(_reason_ops()[op](left, right))
    echo = prompt + call + "<结果>" + result + "</结果>"
    full = call + "<结果>" + result + "</结果>" + result
    return prompt, call, echo, result, full


def reason_batches(tokenizer, chunk=12, keep_holdout=False):
    """1 到 9 的加减乘。默认留出若干格。每题两条：写出思考和调用，以及看见结果后说出数字。"""
    calls = []
    echos = []
    holdouts = []
    for op in ("加", "减", "乘"):
        for left in range(1, 10):
            for right in range(1, 10):
                prompt, call, echo, result, full = reason_example(op, left, right)
                if (op, left, right) in REASON_HOLDOUT:
                    holdouts.append((prompt, full))
                    if not keep_holdout:
                        continue
                calls.append(_prompt_answer(tokenizer, prompt, call))
                echos.append(_prompt_answer(tokenizer, echo, result))
    return _chunks(calls, chunk, tokenizer.pad_id), _chunks(echos, chunk, tokenizer.pad_id), holdouts


# 删掉这些词以后句子更短，剩下的字都还在原句里。
_FILLERS = {
    "the", "a", "an", "of", "and", "to", "very", "really", "that", "which",
    "is", "are", "was", "were", "in", "on", "for", "with", "as", "by", "from",
    "this", "these", "those", "it", "its", "be", "been", "or", "at", "into",
}


def _compress(text):
    """去掉虚词，得到一句更短、用字仍来自原文的回答。"""
    words = text.split()
    if len(words) < 6:
        return None
    kept = []
    for word in words:
        bare = word.lower().strip(".,;:\"'`()[]")
        if bare in _FILLERS:
            continue
        kept.append(word)
    short = " ".join(kept)
    if len(kept) >= 3 and len(short) < len(text):
        return short
    cut = max(3, (len(words) * 2) // 3)
    if cut < len(words):
        return " ".join(words[:cut])
    return None


def shorten_batches(tokenizer, max_seq, limit=240):
    """中文原题穿插在英文句子里。句子按删虚词改短，整段要放得进上下文。"""
    gold = []
    for row in read_jsonl(DATA / "sft.jsonl"):
        gold.append(_prompt_answer(tokenizer, row["prompt"], row["answer"]))
    english = []
    path = HF / "text.txt"
    room = min(max_seq, 72)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            for piece in line.replace("!", ".").replace("?", ".").split(". "):
                content = " ".join(piece.strip(" .").split())
                if len(content) < 40 or len(content) > 120:
                    continue
                answer = _compress(content)
                if answer is None:
                    continue
                prompt = "指令：把下面这句话改短。\n内容：%s\n回答：" % content
                try:
                    ids, labels = _prompt_answer(tokenizer, prompt, answer)
                except KeyError:
                    continue
                if len(ids) > room:
                    continue
                english.append((ids, labels))
                if len(english) >= limit:
                    break
            if len(english) >= limit:
                break
    rows = []
    for index, row in enumerate(english):
        rows.append(row)
        if index % 6 == 0:
            rows.append(gold[index % len(gold)])
    if not rows:
        rows = list(gold)
    return _chunks(rows, 16, tokenizer.pad_id)


def _chunks(rows, chunk, pad_id):
    """把样本按固定条数切开，避免一次把两百条都放进显存。"""
    batches = []
    for start in range(0, len(rows), chunk):
        batches.append(_pad(rows[start : start + chunk], pad_id))
    return batches


def text_hold_batch(tokenizer):
    """改短和对话放进同一批，学算式时用来拽住已经对上的句子。"""
    rows = []
    for name in ("sft.jsonl", "dialogue.jsonl"):
        for row in read_jsonl(DATA / name):
            rows.append(_prompt_answer(tokenizer, row["prompt"], row["answer"]))
    return _pad(rows, tokenizer.pad_id)


def tool_echo_contrast(tokenizer):
    """每种结果数字只留一条，放进同一批。这样「5」和「6」必须分开，不能一律说 5。"""
    rows = []
    seen = set()
    for row in read_jsonl(DATA / "tool.jsonl"):
        steps = row["steps"] if "steps" in row else [{"call": row["call"], "result": row["result"]}]
        prefix = row["prompt"]
        for step in steps:
            prefix = prefix + step["call"] + "<结果>" + step["result"] + "</结果>"
        answer = row["answer"]
        if answer in seen:
            continue
        seen.add(answer)
        rows.append(_prompt_answer(tokenizer, prefix, answer))
    return _pad(rows, tokenizer.pad_id)


def photo_focus_batch(tokenizer, image_size, patch_size, max_seq, answers, copies=2):
    """指定类别各留几张，放进同一批。用来把互相说错的物体拉开。"""
    path = HF / "images.jsonl"
    if not path.exists():
        return None
    n_patches = (image_size // patch_size) ** 2
    rows = []
    images = []
    seen = {answer: 0 for answer in answers}
    for row in read_jsonl(path):
        answer = row["answer"]
        if answer not in seen or seen[answer] >= copies:
            continue
        seen[answer] += 1
        ids, labels = _prompt_answer(tokenizer, row["prompt"], answer)
        room = max_seq - n_patches
        if len(ids) > room:
            ids = ids[:room]
            labels = labels[:room]
        rows.append((ids, labels))
        images.append(_load_rgb(HF / row["file"], image_size))
    batch = _pad(rows, tokenizer.pad_id)
    ignore = torch.full((batch["labels"].size(0), n_patches), -100, dtype=torch.long)
    batch["labels"] = torch.cat((ignore, batch["labels"]), dim=1)
    batch["images"] = torch.stack(images, dim=0)
    return batch


def photo_contrast_batch(tokenizer, image_size, patch_size, max_seq):
    """每一类只留第一张，拼成一个批次。同一步里类别名不同，避免全部说成同一个词。"""
    path = HF / "images.jsonl"
    if not path.exists():
        return None
    n_patches = (image_size // patch_size) ** 2
    rows = []
    images = []
    seen = set()
    for row in read_jsonl(path):
        answer = row["answer"]
        if answer in seen:
            continue
        seen.add(answer)
        ids, labels = _prompt_answer(tokenizer, row["prompt"], answer)
        room = max_seq - n_patches
        if len(ids) > room:
            ids = ids[:room]
            labels = labels[:room]
        rows.append((ids, labels))
        images.append(_load_rgb(HF / row["file"], image_size))
    batch = _pad(rows, tokenizer.pad_id)
    ignore = torch.full((batch["labels"].size(0), n_patches), -100, dtype=torch.long)
    batch["labels"] = torch.cat((ignore, batch["labels"]), dim=1)
    batch["images"] = torch.stack(images, dim=0)
    return batch


def tool_batches(tokenizer):
    """单次加减乘，或连续两步。模型只写调用，结果由程序填进下一条样本。"""
    batches = []
    for row in read_jsonl(DATA / "tool.jsonl"):
        if "steps" in row:
            batches.extend(_tool_trace(tokenizer, row["prompt"], row["steps"], row["answer"]))
        else:
            steps = [{"call": row["call"], "result": row["result"]}]
            batches.extend(_tool_trace(tokenizer, row["prompt"], steps, row["answer"]))
    return batches


def mm_batches(tokenizer, image_size, patch_size):
    """纯色或左右分色。图像块个数要和标签前面补上的 -100 一样长。"""
    batches = []
    n_patches = (image_size // patch_size) ** 2
    for row in read_jsonl(DATA / "mm.jsonl"):
        ids, labels = _prompt_answer(tokenizer, row["prompt"], row["answer"])
        batch = _pad([(ids, labels)], tokenizer.pad_id)
        # 图像块在文字前面，这些位置不计算文字损失。
        ignore = torch.full((1, n_patches), -100, dtype=torch.long)
        batch["labels"] = torch.cat((ignore, batch["labels"]), dim=1)
        batch["images"] = make_image(row, image_size).unsqueeze(0)
        batches.append(batch)
    return batches


def _load_rgb(path, size):
    """读一张图，缩放到模型的边长，像素归一化到 0 到 1。"""
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    # from_numpy 总是落在 CPU。默认设备是显卡时，读完就搬过去。
    device = torch.empty(0).device
    if device.type != "cpu":
        tensor = tensor.to(device)
    return tensor


def _vision_batch(tokenizer, prompt, answer, images, n_tokens, max_seq):
    """图像或视频帧接在文字前。这些位置的标签是 -100，损失只算回答。"""
    ids, labels = _prompt_answer(tokenizer, prompt, answer)
    # 图像块会占掉上下文，文字太长就截到放得下。
    room = max_seq - n_tokens
    if len(ids) > room:
        ids = ids[:room]
        labels = labels[:room]
    batch = _pad([(ids, labels)], tokenizer.pad_id)
    ignore = torch.full((1, n_tokens), -100, dtype=torch.long)
    batch["labels"] = torch.cat((ignore, batch["labels"]), dim=1)
    batch["images"] = images.unsqueeze(0)
    return batch


def hf_text_batches(tokenizer, max_seq):
    """Hugging Face 文本，一行一条，学下一个子词。"""
    path = HF / "text.txt"
    if not path.exists():
        return []
    batches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if len(line) < 20:
            continue
        ids = tokenizer.encode(line, add_bos=True, add_eos=True)[:max_seq]
        batches.append(_pad([(ids, ids)], tokenizer.pad_id))
    return batches


def hf_image_batches(tokenizer, image_size, patch_size, max_seq, per_label=2):
    """CIFAR-10 图片和类别名。每类只留几张，步数才够把名字记下来。"""
    path = HF / "images.jsonl"
    if not path.exists():
        return []
    n_patches = (image_size // patch_size) ** 2
    batches = []
    seen = {}
    for row in read_jsonl(path):
        answer = row["answer"]
        if seen.get(answer, 0) >= per_label:
            continue
        seen[answer] = seen.get(answer, 0) + 1
        image = _load_rgb(HF / row["file"], image_size)
        batches.append(_vision_batch(tokenizer, row["prompt"], row["answer"], image, n_patches, max_seq))
    return batches


def hf_video_batches(tokenizer, image_size, patch_size, max_seq):
    """一个视频抽两帧，按时间接成图像块。回答是动作类别。"""
    path = HF / "videos.jsonl"
    if not path.exists():
        return []
    n_patches = (image_size // patch_size) ** 2
    batches = []
    for row in read_jsonl(path):
        frames = torch.stack([_load_rgb(HF / name, image_size) for name in row["frames"]])
        batches.append(
            _vision_batch(
                tokenizer,
                row["prompt"],
                row["answer"],
                frames,
                n_patches * frames.size(0),
                max_seq,
            )
        )
    return batches
