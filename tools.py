"""工具调用：模型自己决定直接回答，还是写出调用。程序只负责算出结果并接回上下文。"""

from generate import generate

OPS = {
    "加": lambda a, b: a + b,
    "减": lambda a, b: a - b,
    "乘": lambda a, b: a * b,
}


def run_tool(spec):
    """spec 形如「加:2:3」或「乘:2:3」。只做整数加减乘，结果用十进制字符串返回。"""
    name, left, right = spec.split(":")
    return str(OPS[name](int(left), int(right)))


def parse_call(text):
    """取出 `<工具>` 和 `</工具>` 中间的算式。缺一边就当作还没写出调用。"""
    start = text.find("<工具>")
    end = text.find("</工具>")
    if start < 0 or end < 0 or end < start:
        return None
    return text[start + len("<工具>") : end]


def act(model, tokenizer, prompt, image=None, max_calls=2, max_new_tokens=48, temperature=0.0):
    """同一个入口。模型直接写完回答就结束；写出工具调用时，程序算出结果接回去，它再决定要不要调下一次。"""
    pending = prompt
    visible = ""
    for call_index in range(max_calls + 1):
        # 最后一轮不再等工具标记，避免它停在半个调用上。
        stop = "</工具>" if call_index < max_calls else None
        piece = generate(
            model,
            tokenizer,
            pending,
            image=image,
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
        )
        visible += piece
        if call_index >= max_calls:
            return visible
        spec = parse_call(piece)
        if spec is None:
            return visible
        try:
            result = run_tool(spec)
        except (KeyError, ValueError):
            return visible
        observation = "<结果>" + result + "</结果>"
        visible += observation
        pending += piece + observation
    return visible
