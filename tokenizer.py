"""从语料里收集单字，再把相邻高频片段合并成子词。"""

import json
from pathlib import Path


def _merge_symbols(symbols, pair, joined):
    """把词里所有相邻的 pair 收成一个子词。"""
    merged = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(joined)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


class BPETokenizer:
    """先按字切开，再按语料里的相邻次数合并。特殊符号不参与合并。"""

    pad = "<pad>"
    bos = "<bos>"
    eos = "<eos>"

    def __init__(self, stoi, merges):
        self.stoi = dict(stoi)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.merges = [tuple(pair) for pair in merges]

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_id(self):
        return self.stoi[self.pad]

    @property
    def bos_id(self):
        return self.stoi[self.bos]

    @property
    def eos_id(self):
        return self.stoi[self.eos]

    @classmethod
    def fit(cls, texts, num_merges):
        """特殊符号占 0、1、2。随后每轮把出现至少两次的最高频相邻对收成一个新词。"""
        stoi = {cls.pad: 0, cls.bos: 1, cls.eos: 2}
        for text in texts:
            for ch in text:
                if ch not in stoi:
                    stoi[ch] = len(stoi)
        corpus = [list(text) for text in texts if text]
        merges = []
        for _ in range(num_merges):
            counts = {}
            for symbols in corpus:
                for left, right in zip(symbols, symbols[1:]):
                    counts[(left, right)] = counts.get((left, right), 0) + 1
            if not counts:
                break
            pair, times = max(counts.items(), key=lambda item: item[1])
            if times < 2:
                break
            joined = pair[0] + pair[1]
            merges.append(pair)
            stoi[joined] = len(stoi)
            corpus = [_merge_symbols(symbols, pair, joined) for symbols in corpus]
        return cls(stoi, merges)

    def add_chars(self, text):
        """把词表里没有的字补在末尾。已有合并规则不变，新字先按单字用。"""
        added = []
        for ch in text:
            if ch not in self.stoi:
                self.stoi[ch] = len(self.stoi)
                self.itos[self.stoi[ch]] = ch
                added.append(ch)
        return added

    def encode(self, text, add_bos=False, add_eos=False):
        """按学到的顺序合并相邻片段。词表里没有的字直接报错。"""
        symbols = []
        for ch in text:
            if ch not in self.stoi:
                raise KeyError("词表里没有字符 %r，训练前需要把它写进语料" % ch)
            symbols.append(ch)
        for left, right in self.merges:
            symbols = _merge_symbols(symbols, (left, right), left + right)
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for symbol in symbols:
            ids.append(self.stoi[symbol])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """子词直接拼回字符串。填充、开头和结尾符号不出现在结果里。"""
        parts = []
        specials = (self.pad, self.bos, self.eos)
        for i in ids:
            piece = self.itos.get(int(i), "")
            if piece in specials:
                continue
            parts.append(piece)
        return "".join(parts)

    def save(self, path):
        """字表和合并顺序写成 JSON，编号必须和权重一起用。"""
        payload = {"stoi": self.stoi, "merges": [list(pair) for pair in self.merges]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """从 JSON 还原。没有 merges 字段时按纯字词表读。"""
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if "stoi" in blob:
            return cls(blob["stoi"], blob.get("merges", []))
        return cls(blob, [])
