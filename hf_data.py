"""从 Hugging Face 下载一小份文本、图片和视频，存成训练能直接读的文件。"""

import json
import tarfile
from pathlib import Path

import av
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent / "data" / "hf"
TEXT_CAP = 3_000_000
ZH_CAP = 800_000
N_IMAGES = 400
N_VIDEOS = 24

CIFAR_NAMES = {
    0: "飞机",
    1: "汽车",
    2: "鸟",
    3: "猫",
    4: "鹿",
    5: "狗",
    6: "青蛙",
    7: "马",
    8: "船",
    9: "卡车",
}


def _write_text():
    """WikiText-2 是英文维基。截断到几百万字，避免词表拟合扫太久。"""
    lines = []
    used = 0
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for row in wiki:
        text = " ".join(row["text"].split())
        if len(text) < 40:
            continue
        lines.append(text)
        used += len(text)
        if used >= TEXT_CAP:
            break
    zh_used = 0
    try:
        # 中文情感语料很小，用来把汉字补进词表。整库中文维基有几个 GB，这里不下。
        zh = load_dataset("seamew/ChnSentiCorp", split="train")
        for row in zh:
            text = row.get("text") or row.get("sentence") or ""
            text = " ".join(str(text).split())
            if len(text) < 8:
                continue
            lines.append(text)
            zh_used += len(text)
            if zh_used >= ZH_CAP:
                break
    except Exception as exc:
        print("中文语料这次没下下来：%s" % exc)
    path = ROOT / "text.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("文本 %d 行，中文约 %d 字，写到 %s" % (len(lines), zh_used, path))


def _write_images():
    """CIFAR-10 的图和中文类别名。只留前几百张，下载仍会把压缩包取全。"""
    image_dir = ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    data = load_dataset("cifar10", split="train[:%d]" % N_IMAGES)
    for index, row in enumerate(data):
        name = "images/%04d.png" % index
        row["img"].save(image_dir / ("%04d.png" % index))
        label = CIFAR_NAMES[int(row["label"])]
        rows.append(
            {
                "file": name,
                "prompt": "图中的物体是什么？\n回答：",
                "answer": label,
            }
        )
    (ROOT / "images.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print("图片 %d 张" % len(rows))


def _frame_image(video_path, which):
    """取第 which 帧。短视频不够两帧时，后面的帧就重复最后一帧。"""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_image())
        if len(frames) > which:
            break
    container.close()
    if not frames:
        raise RuntimeError("视频没有画面：%s" % video_path)
    return frames[min(which, len(frames) - 1)]


def _write_videos():
    """UCF101 子集大约 170MB。每个类别抽几段，每段只留两帧。"""
    archive = hf_hub_download(
        repo_id="sayakpaul/ucf101-subset",
        filename="UCF101_subset.tar.gz",
        repo_type="dataset",
    )
    extract_to = ROOT / "ucf"
    if not extract_to.exists():
        with tarfile.open(archive) as packed:
            packed.extractall(ROOT)
        extracted = ROOT / "UCF101_subset"
        if extracted.exists():
            extracted.rename(extract_to)
    train_dir = extract_to / "train"
    if not train_dir.exists():
        train_dir = next(extract_to.rglob("train"))
    frame_dir = ROOT / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    videos = sorted(train_dir.rglob("*.avi")) + sorted(train_dir.rglob("*.mp4"))
    per_class = {}
    for video_path in videos:
        label = video_path.parent.name
        if per_class.get(label, 0) >= 3:
            continue
        per_class[label] = per_class.get(label, 0) + 1
        index = len(rows)
        names = []
        for frame_index in (0, 8):
            image = _frame_image(video_path, frame_index).resize((64, 64), Image.BILINEAR)
            name = "frames/%03d_%d.png" % (index, frame_index)
            image.save(frame_dir / ("%03d_%d.png" % (index, frame_index)))
            names.append(name)
        rows.append(
            {
                "frames": names,
                "prompt": "这段视频在做什么？\n回答：",
                "answer": label,
            }
        )
        if len(rows) >= N_VIDEOS:
            break
    (ROOT / "videos.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print("视频 %d 段" % len(rows))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_text()
    _write_images()
    _write_videos()


if __name__ == "__main__":
    main()
