# -*- coding: utf-8 -*-
"""faster-whisper 批量转写工具 - 公共模块

集中放三样东西，供 batch_transcribe.py / fw-demo.py 共用：
  1. 语言代码 -> 中文名映射
  2. 设备探测与模型加载（含 GPU 真机推理探测、缺运行库的可读报错）
  3. 时间戳 / 终端显示宽度等小工具
"""
from __future__ import annotations

import os
import unicodedata

# ---------------------------------------------------------------------------
# 语言表：faster-whisper 返回 ISO 639-1 代码，这里翻成中文名；未收录的原样显示
# ---------------------------------------------------------------------------
LANG_NAMES = {
    "en": "英语", "zh": "中文", "yue": "粤语", "ja": "日语", "ko": "韩语",
    "fr": "法语", "de": "德语", "es": "西班牙语", "pt": "葡萄牙语",
    "it": "意大利语", "ru": "俄语", "ar": "阿拉伯语", "hi": "印地语",
    "bn": "孟加拉语", "ur": "乌尔都语", "vi": "越南语", "th": "泰语",
    "id": "印尼语", "ms": "马来语", "tl": "他加禄语", "nl": "荷兰语",
    "pl": "波兰语", "tr": "土耳其语", "uk": "乌克兰语", "he": "希伯来语",
    "sv": "瑞典语", "no": "挪威语", "da": "丹麦语", "fi": "芬兰语",
    "cs": "捷克语", "el": "希腊语", "hu": "匈牙利语", "ro": "罗马尼亚语",
    "fa": "波斯语", "ta": "泰米尔语", "te": "泰卢固语", "kn": "卡纳达语",
    "ml": "马拉雅拉姆语", "si": "僧伽罗语", "ne": "尼泊尔语", "km": "高棉语",
    "lo": "老挝语", "my": "缅甸语", "mn": "蒙古语", "kk": "哈萨克语",
    "uz": "乌兹别克语", "az": "阿塞拜疆语", "ka": "格鲁吉亚语", "hy": "亚美尼亚语",
    "sw": "斯瓦希里语", "af": "南非荷兰语", "is": "冰岛语", "et": "爱沙尼亚语",
    "lv": "拉脱维亚语", "lt": "立陶宛语", "sl": "斯洛文尼亚语", "sk": "斯洛伐克语",
    "hr": "克罗地亚语", "sr": "塞尔维亚语", "bg": "保加利亚语", "mk": "马其顿语",
    "sq": "阿尔巴尼亚语", "bs": "波斯尼亚语", "ca": "加泰罗尼亚语", "eu": "巴斯克语",
    "gl": "加利西亚语", "cy": "威尔士语", "mt": "马耳他语", "ha": "豪萨语",
    "am": "阿姆哈拉语", "jw": "爪哇语", "su": "巽他语", "pa": "旁遮普语",
    "gu": "古吉拉特语", "or": "奥里亚语", "as": "阿萨姆语",
}

# 模型体积（仅用于首次下载前的提示，取自官方仓库大小，四舍五入）
MODEL_SIZES = {
    "tiny": "约 75MB", "tiny.en": "约 75MB",
    "base": "约 145MB", "base.en": "约 145MB",
    "small": "约 460MB", "small.en": "约 460MB",
    "medium": "约 1.5GB", "medium.en": "约 1.5GB",
    "large-v1": "约 3.0GB", "large-v2": "约 3.0GB",
    "large-v3": "约 3.1GB", "large": "约 3.1GB",
    "large-v3-turbo": "约 1.6GB", "turbo": "约 1.6GB",
    "distil-large-v3": "约 1.5GB",
}

GPU_LIB_HINT = (
    "  缺少 CUDA 12 运行库。任选其一：\n"
    "    1) pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
    "    2) 手动把 cublas64_12.dll、cudnn_ops_infer64_9.dll 等放进 python\\ 目录\n"
    "    3) 先跑 CPU：加参数 --device cpu"
)


def lang_name(code: str) -> str:
    """语言代码 -> 中文名（未收录时原样返回代码）"""
    return LANG_NAMES.get(code, code)


def fmt_time(seconds: float) -> str:
    """秒 -> SRT 时间戳 00:00:00,000"""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def disp_width(text: str) -> int:
    """终端显示宽度：中日韩全角字符占 2 列"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int) -> str:
    """按显示宽度左对齐补空格（修中文列宽错位）"""
    return text + " " * max(0, width - disp_width(text))


def truncate_disp(text: str, width: int, tail: str = "…") -> str:
    """按显示宽度截断，超长补省略号"""
    if disp_width(text) <= width:
        return text
    limit = width - disp_width(tail)
    out, used = [], 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + w > limit:
            break
        out.append(ch)
        used += w
    return "".join(out) + tail


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def is_local_model(model: str) -> bool:
    """传入的是本地模型目录而不是模型名"""
    return os.path.isdir(model)


def _hf_repo_id(model: str):
    """模型名 -> HuggingFace 仓库 id（取不到返回 None）"""
    try:
        from faster_whisper.utils import _MODELS
    except Exception:
        return None
    return _MODELS.get(model)


def model_is_cached(model: str) -> bool:
    """判断模型是否已经在本地 HuggingFace 缓存里（避免误报"要下载"）"""
    repo = _hf_repo_id(model)
    if not repo:
        return False
    try:
        from huggingface_hub import constants as hf_constants
    except Exception:
        return False
    base = os.environ.get("HF_HUB_CACHE") or getattr(hf_constants, "HF_HUB_CACHE", "")
    if not base or not os.path.isdir(base):
        return False
    return os.path.isdir(os.path.join(base, "models--" + repo.replace("/", "--")))


def announce_model(model: str) -> None:
    """加载前提示：这是本地目录 / 已缓存 / 需要联网下载"""
    if is_local_model(model):
        print(f"[模型] 使用本地模型目录: {os.path.abspath(model)}")
        return

    size = MODEL_SIZES.get(model, "")
    if model_is_cached(model):
        print(f"[模型] {model} 已在本地缓存，直接加载")
        return

    repo = _hf_repo_id(model)
    if repo is None:
        print(f"[模型] 未识别的模型名 '{model}'，将尝试按原样加载；"
              f"若失败请改用本地模型目录路径。")
        return

    print("=" * 60)
    print(f"[模型] 本地没有 {model}，首次运行需要联网下载（{size}）。")
    print(f"       来源: https://huggingface.co/{repo}")
    print("       下载期间没有进度条、会静默等待，属于正常现象，不要关闭窗口。")
    print("       国内网络建议先设镜像再运行：")
    print("         Windows CMD:  set HF_ENDPOINT=https://hf-mirror.com")
    print("         PowerShell :  $env:HF_ENDPOINT='https://hf-mirror.com'")
    print("       已有离线模型：直接用 --model 指向模型目录，例如")
    print("         --model models\\large-v3")
    print("=" * 60, flush=True)


def _gpu_hint_message(exc: Exception) -> str:
    msg = str(exc)
    lowered = msg.lower()
    if "cublas" in lowered or "cudnn" in lowered or "cuda" in lowered:
        return GPU_LIB_HINT
    return ""


def load_model(model: str, device: str = "auto"):
    """加载模型，返回 (model, 设备标识)。

    device="auto"：先探测 GPU，再用一个空特征矩阵做**真实推理**，
    确认 cublas/cudnn 真的能加载；失败自动回退 CPU(int8)。
    device="cuda"：强制 GPU，失败时抛出带修复建议的 RuntimeError。
    """
    import numpy as np
    from faster_whisper import WhisperModel

    assert device in ("auto", "cpu", "cuda"), device

    if device in ("auto", "cuda"):
        try:
            import ctranslate2

            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception as exc:  # 未装 ctranslate2 或驱动异常
            cuda_count = 0
            if device == "cuda":
                raise RuntimeError(f"无法检测 CUDA 设备: {exc}") from exc

        if cuda_count > 0:
            if device == "cuda":
                print(f"检测到 {cuda_count} 个 CUDA 设备，强制使用 GPU ...")
            else:
                print(f"检测到 {cuda_count} 个 CUDA 设备，尝试 GPU 加速 ...")
            try:
                m = WhisperModel(model, device="cuda", compute_type="float16")
                # 真实推理探测：确认 cublas/cudnn 运行库可用，而不是只建了对象
                m.encode(np.zeros((80, 3000), dtype="float32"))
                print("GPU 加速可用 (cuda/float16)")
                return m, "cuda/float16"
            except Exception as exc:
                hint = _gpu_hint_message(exc)
                if device == "cuda":
                    raise RuntimeError(
                        f"GPU 不可用: {exc}\n{hint}" if hint else f"GPU 不可用: {exc}"
                    ) from exc
                print(f"GPU 不可用，回退 CPU。原因: {exc}")
                if hint:
                    print(hint)
        elif device == "cuda":
            raise RuntimeError("没有检测到可用的 CUDA 设备，请去掉 --device cuda 或改用 --device cpu")

    print("使用 CPU 模式 (int8) ...")
    return WhisperModel(model, device="cpu", compute_type="int8"), "cpu/int8"


def warn_if_slow_on_cpu(device_tag: str, model: str) -> None:
    """大模型在 CPU 上会非常慢，提前打招呼"""
    if device_tag != "cpu/int8":
        return
    heavy = {"large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo"}
    name = os.path.basename(os.path.normpath(model)).lower() if is_local_model(model) else model
    if model in heavy or any(k in name for k in ("large", "turbo")):
        print("提示: 当前是 CPU 模式 + 大模型，速度约为实时的 0.3~1 倍。")
        print("      想快很多：修好 GPU（见上）/ 换 --model medium / 换 --model small。")
