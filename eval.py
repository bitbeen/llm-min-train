"""用同一套题对比权重：九九乘法、500 条文本、50 张图片。"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from config import Config
from data import DATA, HF, _load_rgb, read_jsonl
from generate import pick_device
from model import TinyLM
from tokenizer import BPETokenizer
from tools import act

ROOT = Path(__file__).resolve().parent


def load_from(directory):
    """按目录里保存的配置把权重放到显卡上。"""
    directory = Path(directory)
    tokenizer = BPETokenizer.load(directory / "tokenizer.json")
    blob = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    cfg = Config(**blob["config"])
    pick_device()
    model = TinyLM(cfg, tokenizer.vocab_size)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, tokenizer, cfg


def _params(model):
    return sum(param.numel() for param in model.parameters())


def eval_table(model, tokenizer):
    """1 到 9 的乘法。得数看结果后面的数字，整段要和训练时的工具格式一致。"""
    number_ok = 0
    exact_ok = 0
    misses = []
    for left in range(1, 10):
        for right in range(1, 10):
            product = left * right
            prompt = "算式%d乘%d。\n回答：" % (left, right)
            pred = act(model, tokenizer, prompt)
            gold = "<思考>先取%d乘%d</思考><工具>乘:%d:%d</工具><结果>%d</结果>%d" % (
                left,
                right,
                left,
                right,
                product,
                product,
            )
            tail = pred.split("</结果>")[-1].strip() if "</结果>" in pred else pred.strip()
            number_hit = tail == str(product)
            exact_hit = pred == gold
            number_ok += int(number_hit)
            exact_ok += int(exact_hit)
            if not number_hit and len(misses) < 8:
                misses.append("%d×%d 期望 %d 得到 %s" % (left, right, product, pred.replace("\n", "")[:40]))
    return number_ok, exact_ok, 81, misses


def _shorten_prompt(content):
    return "指令：把下面这句话改短。\n内容：%s\n回答：" % content


def _is_shortened(content, pred):
    """比原句短，而且用的是原句里的字，才算改短成功。"""
    pred = pred.strip()
    if len(pred) < 2 or len(pred) >= len(content):
        return False
    if "指令" in pred or "回答" in pred:
        return False
    kept = sum(1 for ch in pred if ch in content or ch.isspace())
    return kept / len(pred) >= 0.8


def _passages(tokenizer, cfg, limit):
    """从语料里抽能放进上下文的句子，凑满 500 段。"""
    found = []
    seen = set()
    for line in (HF / "text.txt").read_text(encoding="utf-8").splitlines():
        for piece in line.replace("!", ".").replace("?", ".").split(". "):
            content = " ".join(piece.strip(" .").split())
            if len(content) < 40 or len(content) > 160 or content in seen:
                continue
            prompt = _shorten_prompt(content)
            try:
                ids = tokenizer.encode(prompt)
            except KeyError:
                continue
            if len(ids) >= cfg.max_seq - 8:
                continue
            seen.add(content)
            found.append(content)
            if len(found) >= limit:
                return found
    return found


def eval_text(model, tokenizer, cfg, limit):
    """500 段都用改短指令。成功的标准是回答更短，并且字来自原句。"""
    passages = _passages(tokenizer, cfg, limit)
    ok = 0
    samples = []
    for index, content in enumerate(passages, start=1):
        pred = act(model, tokenizer, _shorten_prompt(content))
        hit = _is_shortened(content, pred)
        ok += int(hit)
        if len(samples) < 6:
            samples.append((content[:48], pred.replace("\n", "")[:48], hit))
        if index == 1 or index == len(passages) or index % 100 == 0:
            print("  改短进度 %d/%d  目前 %d" % (index, len(passages), ok), flush=True)
    labeled = 0
    labeled_n = 0
    for row in read_jsonl(DATA / "sft.jsonl"):
        labeled_n += 1
        labeled += int(act(model, tokenizer, row["prompt"]) == row["answer"])
    return ok, len(passages), labeled, labeled_n, samples


def eval_images(model, tokenizer, cfg, limit):
    """每类先取够，一共 50 张。回答和类别名完全一致才算对。"""
    picked = []
    seen = Counter()
    per_class = max(1, limit // 10)
    for row in read_jsonl(HF / "images.jsonl"):
        if seen[row["answer"]] >= per_class:
            continue
        seen[row["answer"]] += 1
        picked.append(row)
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        used = {row["file"] for row in picked}
        for row in read_jsonl(HF / "images.jsonl"):
            if row["file"] in used:
                continue
            picked.append(row)
            if len(picked) >= limit:
                break
    ok = 0
    by_class = Counter()
    hit_class = Counter()
    misses = []
    for row in picked:
        photo = _load_rgb(HF / row["file"], cfg.image_size)
        pred = act(model, tokenizer, row["prompt"], image=photo)
        by_class[row["answer"]] += 1
        if pred == row["answer"]:
            ok += 1
            hit_class[row["answer"]] += 1
        elif len(misses) < 8:
            misses.append("%s => %s" % (row["answer"], pred.replace("\n", "")[:24]))
    return ok, len(picked), by_class, hit_class, misses


def report(directory):
    """打印一份权重在三组题上的结果。"""
    model, tokenizer, cfg = load_from(directory)
    params = _params(model)
    device = next(model.parameters()).device
    print("== %s ==" % directory)
    print("参数量 %d  设备 %s" % (params, device))

    number_ok, exact_ok, table_n, table_miss = eval_table(model, tokenizer)
    print("九九得数 %d/%d  整段 %d/%d" % (number_ok, table_n, exact_ok, table_n))
    for line in table_miss:
        print("  " + line)

    ok, lines, labeled, labeled_n, samples = eval_text(model, tokenizer, cfg, 500)
    print("改短 %d/%d  学过的原题 %d/%d" % (ok, lines, labeled, labeled_n))
    for content, pred, hit in samples:
        print("  %s | %s => %s" % ("对" if hit else "错", content, pred))

    image_ok, image_n, by_class, hit_class, image_miss = eval_images(model, tokenizer, cfg, 50)
    print("图片 %d/%d" % (image_ok, image_n))
    names = sorted(by_class)
    print("  " + " ".join("%s %d/%d" % (name, hit_class[name], by_class[name]) for name in names))
    for line in image_miss:
        print("  " + line)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "params": params,
        "table": (number_ok, table_n),
        "table_exact": (exact_ok, table_n),
        "text": (ok, lines, labeled, labeled_n),
        "image": (image_ok, image_n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", help="放着 model.pt 和 tokenizer.json 的目录")
    args = parser.parse_args()
    directories = [Path(item) for item in args.dirs] if args.dirs else []
    if not directories:
        for name in ("out-0.1b", "out"):
            path = ROOT / name
            if (path / "model.pt").exists():
                directories.append(path)
    if not directories:
        raise SystemExit("没有找到 model.pt")
    rows = [(path, report(path)) for path in directories]
    if len(rows) < 2:
        return
    print("== 对比 ==")
    for path, row in rows:
        text_ok, text_n, labeled, labeled_n = row["text"]
        table_ok, table_n = row["table"]
        image_ok, image_n = row["image"]
        print(
            "%s  参数 %.2f亿  九九 %d/%d  改短 %d/%d  原题 %d/%d  图片 %d/%d"
            % (
                path.name,
                row["params"] / 1e8,
                table_ok,
                table_n,
                text_ok,
                text_n,
                labeled,
                labeled_n,
                image_ok,
                image_n,
            )
        )


if __name__ == "__main__":
    main()
