# -*- coding: utf-8 -*-
"""faster-whisper 快速启动示例：转写 jfk.flac"""
import sys, time

import ctranslate2
from faster_whisper import WhisperModel

_LANG_NAMES = {
    "en": "英语", "zh": "中文", "yue": "粤语", "ja": "日语", "ko": "韩语",
    "fr": "法语", "de": "德语", "es": "西班牙语", "pt": "葡萄牙语",
    "it": "意大利语", "ru": "俄语", "ar": "阿拉伯语", "hi": "印地语",
    "vi": "越南语", "th": "泰语", "id": "印尼语", "ms": "马来语",
    "nl": "荷兰语", "pl": "波兰语", "tr": "土耳其语", "uk": "乌克兰语",
    "he": "希伯来语", "sv": "瑞典语", "el": "希腊语", "fa": "波斯语",
    "ta": "泰米尔语", "ne": "尼泊尔语", "km": "高棉语", "my": "缅甸语",
    "kk": "哈萨克语", "sw": "斯瓦希里语", "ca": "加泰罗尼亚语", "bg": "保加利亚语",
}

def lang_name(code):
    return _LANG_NAMES.get(code, code)

def main():
    # 模型大小/路径：tiny / base / small / ... 或本地模型目录
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"

    # 设备自动检测：有 NVIDIA 显卡用 GPU，否则 CPU
    device, compute_type = "cpu", "int8"
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            device, compute_type = "cuda", "float16"
    except Exception:
        pass

    print(f"[1/3] 加载模型 {model_size} ({device}/{compute_type}) ...")
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"      模型加载完成，耗时 {time.time()-t0:.1f}s")

    audio = sys.argv[1] if len(sys.argv) > 1 else "test1.flac"
    print(f"[2/3] 转写 {audio} ...")
    t0 = time.time()
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,        # 自动过滤静音
        language=None,          # None=自动检测
    )
    print(f"      检测语言: {lang_name(info.language)} ({info.language})  置信度: {info.language_probability:.1%} | "
          f"原始 {info.duration:.1f}s / VAD后 {info.duration_after_vad:.1f}s")

    print("[3/3] 识别结果：")
    for seg in segments:
        print(f"  [{seg.start:7.2f}s -> {seg.end:7.2f}s] {seg.text.strip()}")

    print(f"转写耗时 {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
