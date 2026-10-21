# -*- coding: utf-8 -*-
"""faster-whisper 单文件快速转写

用法:
  python fw-demo.py <音频/视频文件> [模型名或模型目录] [--srt 输出.srt] [--device auto|cpu|cuda]

示例:
  python fw-demo.py 音频.mp4
  python fw-demo.py 音频.mp4 medium
  python fw-demo.py 音频.mp4 models\\large-v3 --srt out.srt

与 batch_transcribe.py 共用 whisper_common.load_model，
因此同样具备 GPU 真机推理探测（缺 cublas/cudnn 时自动回退 CPU 而不是崩在转写阶段）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 嵌入式 Python（python313._pth）不会把脚本目录进 sys.path，便携版里必须先补
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_common import (  # noqa: E402  (必须在 sys.path 补齐之后导入)
    announce_model,
    fmt_time,
    lang_name,
    load_model,
    warn_if_slow_on_cpu,
)

DEFAULT_MODEL = "large-v3"


def write_srt(srt_path: str, lines) -> None:
    with open(srt_path, "w", encoding="utf-8", newline="\n") as f:
        for i, (start, end, text) in enumerate(lines, 1):
            f.write(f"{i}\n{fmt_time(start)} --> {fmt_time(end)}\n{text}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="faster-whisper 单文件转写（默认模型 large-v3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python fw-demo.py 音频.mp4\n"
               "  python fw-demo.py 音频.mp4 medium\n"
               "  python fw-demo.py 音频.mp4 models\\large-v3 --srt out.srt\n",
    )
    parser.add_argument("audio", help="音频/视频文件路径")
    # 兼容老用法：第二个位置参数当模型（便携版启动器就是这么传的）
    parser.add_argument("model_pos", nargs="?", help="(可选) 模型名或模型目录")
    parser.add_argument("--model", dest="model_opt", help="模型名或模型目录，优先级高于位置参数")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="默认 auto")
    parser.add_argument("--srt", help="可选：导出单文件 SRT 到该路径")
    parser.add_argument("--quiet", action="store_true", help="不打印逐句文本")
    args = parser.parse_args()

    audio = args.audio
    if not os.path.isfile(audio):
        print(f"文件不存在: {audio}")
        print("提示: 直接把音频/视频文件拖到脚本图标上，或用 --help 看用法。")
        return 1

    model_arg = args.model_opt or args.model_pos or DEFAULT_MODEL

    print(f"[1/3] 准备模型 {model_arg} ...")
    announce_model(model_arg)
    t0 = time.time()
    try:
        model, device_tag = load_model(model_arg, args.device)
    except Exception as exc:
        print(f"\n模型加载失败: {exc}")
        return 2
    print(f"      模型加载完成 ({device_tag})，耗时 {time.time()-t0:.1f}s")
    warn_if_slow_on_cpu(device_tag, model_arg)

    print(f"[2/3] 转写 {os.path.basename(audio)} ...")
    t0 = time.time()
    try:
        segments, info = model.transcribe(audio, beam_size=5, vad_filter=True, language=None)
    except Exception as exc:
        print(f"转写失败: {exc}")
        return 3

    print(f"      检测语言: {lang_name(info.language)} ({info.language})  "
          f"置信度: {info.language_probability:.1%} | "
          f"原始 {info.duration:.1f}s / VAD后 {info.duration_after_vad:.1f}s")

    print("[3/3] 识别结果：")
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if not args.quiet:
            print(f"  [{seg.start:7.2f}s -> {seg.end:7.2f}s] {text}")
        lines.append((seg.start, seg.end, text))

    if not lines:
        print("  (未检测到语音，VAD 后有效语音时长为 0)")

    print(f"转写耗时 {time.time()-t0:.1f}s")

    if args.srt:
        out_dir = os.path.dirname(os.path.abspath(args.srt))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        write_srt(args.srt, lines)
        print(f"字幕已保存: {os.path.abspath(args.srt)}（{len(lines)} 条）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
