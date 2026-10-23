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
from typing import Optional

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


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda failed with error out of memory" in text


def _is_host_oom(exc: Exception) -> bool:
    """主机内存不足（不是显存）。CT2 会报 mkl_malloc / failed to allocate memory"""
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "mkl_malloc" in text or "failed to allocate memory" in text


def available_ram_gb():
    """当前可用物理内存(GB)，取不到返回 None"""
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024 ** 3)
    except Exception:
        pass
    return None


def _model_disk_gb(model: str):
    """本地模型目录的权重大小(GB)，用于估算内存需求"""
    if is_local_model(model):
        bin_path = os.path.join(model, "model.bin")
        if os.path.isfile(bin_path):
            return os.path.getsize(bin_path) / (1024 ** 3)
        total = 0
        for root, _, names in os.walk(model):
            for n in names:
                try:
                    total += os.path.getsize(os.path.join(root, n))
                except OSError:
                    pass
        return total / (1024 ** 3) if total else None
    guess = {
        "large-v3": 3.1, "large": 3.1, "large-v2": 3.0, "large-v1": 3.0,
        "large-v3-turbo": 1.6, "turbo": 1.6, "medium": 1.5, "medium.en": 1.5,
        "small": 0.46, "small.en": 0.46, "base": 0.15, "tiny": 0.08,
    }
    return guess.get(model)


def preflight_memory(model: str) -> None:
    """加载前看一眼主机内存，不够就提前警告（比等到 mkl_malloc 崩掉友好）"""
    need = _model_disk_gb(model)
    avail = available_ram_gb()
    if not need or not avail:
        return
    if avail < need * 1.8:
        print(f"⚠ 主机可用内存只剩 {avail:.1f}GB，而该模型权重约 {need:.1f}GB。")
        print("  大模型加载需要把权重读进内存，内存不足会报 mkl_malloc / out of memory。")
        print("  建议：关掉浏览器等占内存的程序，或改用 --model medium（约 1.5GB）。")


def _n_mels_of(model) -> Optional[int]:
    """取模型期望的 mel 维数。large-v3 是 128，small/medium 是 80，
    探测矩阵用错维数会把能用的模型误判成不可用。"""
    for obj in (getattr(model, "model", None),
                getattr(getattr(model, "model", None), "model", None)):
        value = getattr(obj, "n_mels", None)
        if isinstance(value, int) and 0 < value <= 256:
            return value
    return None


def _probe(model, np) -> None:
    """空特征推理，确认运行库/显存真的能用。维数自动适配（80 / 128）。"""
    candidates = []
    detected = _n_mels_of(model)
    if detected:
        candidates.append(detected)
    for fallback in (80, 128):
        if fallback not in candidates:
            candidates.append(fallback)

    last_shape_err = None
    for n_mels in candidates:
        try:
            model.encode(np.zeros((n_mels, 3000), dtype="float32"))
            return
        except Exception as exc:
            text = str(exc).lower()
            if "shape" in text or "expected an input" in text:
                last_shape_err = exc   # 维数不对，换下一个再试
                continue
            raise                       # 显存/运行库问题，直接上报
    if last_shape_err:
        raise last_shape_err


def load_model(model: str, device: str = "auto"):
    """加载模型，返回 (model, 设备标识)。

    device="auto"：先探测 GPU，再用一个空特征矩阵做**真实推理**，
    确认 cublas/cudnn 真的能加载；GPU 不可用则回退 CPU(int8)。
    device="cuda"：强制 GPU，全部尝试失败时抛带修复建议的 RuntimeError。

    GPU 上按 float16 -> int8_float16 -> int8 依次降级：
    large-v3 的 float16 权重约 3.1GB，8G 显卡若已被桌面程序占掉一半就会 OOM，
    降成 int8_float16 权重大约减半，仍能跑，精度只掉一点点。
    """
    import numpy as np
    from faster_whisper import WhisperModel

    assert device in ("auto", "cpu", "cuda"), device

    preflight_memory(model)

    def _try(dev: str, compute_type: str):
        m = WhisperModel(model, device=dev, compute_type=compute_type)
        # 真实推理探测：确认运行库真的能加载，而不是只建了对象
        _probe(m, np)
        return m

    if device in ("auto", "cuda"):
        try:
            import ctranslate2

            cuda_count = ctranslate2.get_cuda_device_count()
        except Exception as exc:  # 未装 ctranslate2 或驱动异常
            cuda_count = 0
            if device == "cuda":
                raise RuntimeError(f"无法检测 CUDA 设备: {exc}") from exc

        if cuda_count > 0:
            print(f"检测到 {cuda_count} 个 CUDA 设备，"
                  f"{'强制使用 GPU' if device == 'cuda' else '尝试 GPU 加速'} ...")
            errors = []
            for compute_type in ("float16", "int8_float16", "int8"):
                try:
                    m = _try("cuda", compute_type)
                except Exception as exc:
                    errors.append((compute_type, exc))
                    if _is_host_oom(exc):
                        print(f"主机内存不足，无法加载权重（{compute_type}），继续尝试更省内存的精度 ...")
                    elif _is_oom(exc):
                        print(f"显存不足，{compute_type} 放不下" +
                              ("，降级重试 ..." if compute_type != "int8" else " ..."))
                    continue
                if compute_type != "float16":
                    print(f"GPU 加速可用 (cuda/{compute_type})，"
                          f"已从 float16 降级（显存/内存吃紧，精度影响很小）")
                else:
                    print("GPU 加速可用 (cuda/float16)")
                return m, f"cuda/{compute_type}"

            # GPU 全试失败
            last_exc = errors[-1][1] if errors else RuntimeError("未知错误")
            hint = _gpu_hint_message(last_exc)
            detail = "；".join(f"{ct}: {e}" for ct, e in errors)
            if device == "cuda":
                raise RuntimeError(
                    "GPU 不可用:\n  " + detail + ("\n" + hint if hint else "")
                ) from last_exc
            print(f"GPU 不可用，回退 CPU。原因: {last_exc}")
            if hint:
                print(hint)
        elif device == "cuda":
            raise RuntimeError("没有检测到可用的 CUDA 设备，请去掉 --device cuda 或改用 --device cpu")

    print("使用 CPU 模式 (int8) ...")
    try:
        return _try("cpu", "int8"), "cpu/int8"
    except Exception as exc:
        if _is_host_oom(exc):
            avail = available_ram_gb()
            need = _model_disk_gb(model)
            raise RuntimeError(
                f"主机内存不足，模型加载失败：{exc}\n"
                f"  当前可用内存 {avail:.1f}GB，模型权重约 {need:.1f}GB。\n"
                "  处理办法：关闭浏览器/剪辑软件等占内存的程序，"
                "或改用更小的模型（--model medium / small）。"
                if avail and need else f"主机内存不足，模型加载失败：{exc}"
            ) from exc
        print(f"CPU int8 不可用({exc})，改用 float32 ...")
        try:
            return _try("cpu", "float32"), "cpu/float32"
        except Exception as exc2:
            raise RuntimeError(f"CPU 加载失败: {exc2}") from exc2


def sibling_models(model: str, smaller_than: bool = True) -> list:
    """同一 models 目录下的其他模型目录，按权重大小从大到小返回。

    smaller_than=True 时只返回比当前模型更小的（降级用），
    免得 medium 加载失败却跑去试更大的 large-v3。
    """
    if not is_local_model(model):
        return []
    target = os.path.abspath(model)
    parent = os.path.dirname(target)

    def _bin_size(path: str) -> int:
        try:
            return os.path.getsize(os.path.join(path, "model.bin"))
        except OSError:
            return -1

    current_size = _bin_size(target)
    found = []
    try:
        for name in os.listdir(parent):
            path = os.path.join(parent, name)
            if path == target or not os.path.isfile(os.path.join(path, "model.bin")):
                continue
            size = _bin_size(path)
            if smaller_than and current_size > 0 and size >= current_size:
                continue
            found.append((size, path))
    except OSError:
        return []
    return [path for _, path in sorted(found, reverse=True)]


def warn_if_slow_on_cpu(device_tag: str, model: str) -> None:
    """大模型在 CPU 上会非常慢，提前打招呼"""
    if not device_tag.startswith("cpu/"):
        return
    heavy = {"large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo"}
    name = os.path.basename(os.path.normpath(model)).lower() if is_local_model(model) else model
    if model in heavy or any(k in name for k in ("large", "turbo")):
        print("提示: 当前是 CPU 模式 + 大模型，速度约为实时的 0.3~1 倍。")
        print("      想快很多：修好 GPU（见上）/ 换 --model medium / 换 --model small。")
