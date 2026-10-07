"""加载已保存的权重并生成。训练脚本也会调用这里。"""

from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config
from model import TinyLM
from tokenizer import BPETokenizer

# 权重和训练脚本放在同一处的 out/。
OUT = Path(__file__).resolve().parent / "out"


def pick_device():
    """有显卡就用显卡。之后新建的张量默认落在这台设备上。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        # 32 位矩阵走 TF32，5090 的张量核才参与计算。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    torch.set_default_device(device)
    return device


def place(model):
    """把模型放到 pick_device 选中的设备上。"""
    return model.to(pick_device())


@torch.no_grad()
def generate(model, tokenizer, prompt, image=None, max_new_tokens=24, stop=None, temperature=0.0):
    """temperature 为 0 时取概率最高的字。大于 0 时按这个温度采样。工具调用靠 stop 停在 `</工具>`。"""
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    images = None
    n_image = 0
    if image is not None:
        # 单张图是 (3, H, W)，视频两帧是 (帧数, 3, H, W)。
        images = image.unsqueeze(0).to(device)
        frames = image.size(0) if image.dim() == 4 else 1
        n_image = model.vision.n_patches * frames
    produced = []
    for _ in range(max_new_tokens):
        if input_ids.size(1) + n_image >= model.cfg.max_seq:
            break
        key_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        logits, _, _ = model(input_ids, images=images, key_padding_mask=key_padding_mask)
        scores = logits[0, -1]
        if temperature > 0:
            scores = scores.clone()
            scores[tokenizer.pad_id] = -1e9
            scores[tokenizer.bos_id] = -1e9
            next_id = int(torch.multinomial(F.softmax(scores / temperature, dim=-1), 1))
        else:
            next_id = int(scores.argmax())
        if next_id == tokenizer.eos_id:
            break
        produced.append(next_id)
        text = tokenizer.decode(produced)
        if stop is not None and stop in text:
            return text[: text.index(stop) + len(stop)]
        input_ids = torch.cat((input_ids, torch.tensor([[next_id]], device=input_ids.device)), dim=1)
    return tokenizer.decode(produced)


def load():
    """从本目录的 out/model.pt 读词表和权重。"""
    tokenizer = BPETokenizer.load(OUT / "tokenizer.json")
    blob = torch.load(OUT / "model.pt", map_location="cpu", weights_only=False)
    cfg = Config(**blob["config"])
    pick_device()
    model = TinyLM(cfg, tokenizer.vocab_size)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, tokenizer, cfg


if __name__ == "__main__":
    from data import HF, _load_rgb, read_jsonl
    from tools import act
    from vision import solid_image

    model, tokenizer, cfg = load()
    model = place(model)
    texts = [
        "指令：把下面这句话改短。\n内容：今天天气非常好，很适合出门散步。\n回答：",
        "指令：把下面这句话改短。\n内容：这部电影的剧情非常精彩，演员的表演也很出色。\n回答：",
        "问：今天天气怎么样？\n答：今天天气很好。\n问：适合做什么？\n答：",
    ]
    for prompt in texts:
        print("文本:", prompt.replace("\n", " / "))
        print("回答:", act(model, tokenizer, prompt))
    for color in ("红", "绿", "蓝"):
        image = solid_image(color, cfg.image_size)
        print("色块 %s:" % color, act(model, tokenizer, "这张图是什么颜色？\n回答：", image=image))
    # 每种类别留一张，问题和训练时一样。
    seen = set()
    for row in read_jsonl(HF / "images.jsonl"):
        if row["answer"] in seen:
            continue
        seen.add(row["answer"])
        image = _load_rgb(HF / row["file"], cfg.image_size)
        print("图片 %s:" % row["answer"], act(model, tokenizer, row["prompt"], image=image))
        if len(seen) >= 5:
            break

