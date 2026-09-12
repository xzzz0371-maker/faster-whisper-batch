# -*- coding: utf-8 -*-
"""faster-whisper 批量转写脚本
用法:
  python batch_transcribe.py <文件夹或文件> [模型名] [--srt 输出目录]
示例:
  python batch_transcribe.py D:\我的音频 small
  python batch_transcribe.py D:\我的音频 small --srt D:\字幕
说明:
  - 遍历文件夹下所有音视频文件(递归)，逐个转写
  - 每个文件自动检测语言，直接告诉你是什么语言(中文名+代码+置信度)
  - --srt 目录: 同时把每个文件导出为同名 .srt 字幕
"""
import argparse
import os
import sys
import time

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel


def load_model(model_path, device="auto"):
    """加载模型；device=auto 时检测 GPU 是否真正可用（含 CUDA 运行库），否则回退 CPU"""
    if device == "auto":
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                print("检测到 NVIDIA 显卡，尝试 GPU 加速...")
                m = WhisperModel(model_path, device="cuda", compute_type="float16")
                # 真实推理探测：确认 cublas/cudnn 运行库可用
                m.encode(np.zeros((80, 3000), dtype="float32"))
                print("GPU 加速可用 (float16)")
                return m, "cuda/float16"
        except Exception as e:
            print(f"GPU 不可用 ({e})，回退 CPU ...")
        print("使用 CPU 模式 (int8) ...")
        return WhisperModel(model_path, device="cpu", compute_type="int8"), "cpu/int8"
    if device == "cuda":
        return WhisperModel(model_path, device="cuda", compute_type="float16"), "cuda/float16"
    return WhisperModel(model_path, device="cpu", compute_type="int8"), "cpu/int8"

# 常见语言代码 -> 中文名（faster-whisper 返回 ISO 639-1 代码）
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

AUDIO_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".wmv",
)


def lang_name(code):
    """语言代码 -> 中文名"""
    return LANG_NAMES.get(code, code)


def fmt_time(seconds):
    """秒 -> 字幕时间戳格式 00:00:00,000"""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def collect_files(paths):
    """收集待转写文件：可传多个文件或文件夹(递归)，自动去重"""
    files, seen = [], set()
    for path in paths:
        if os.path.isfile(path):
            real = os.path.abspath(path)
            if real not in seen:
                seen.add(real)
                files.append(real)
        elif os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in names:
                    if name.lower().endswith(AUDIO_EXTS):
                        real = os.path.join(root, name)
                        if real not in seen:
                            seen.add(real)
                            files.append(real)
        else:
            print(f"路径不存在(已跳过): {path}")
    files.sort()
    return files


def transcribe_one(model, filepath, srt_dir):
    """转写单个文件，返回 (语言中文名, 语言代码, 概率, 时长, 耗时, 状态)"""
    t0 = time.time()
    segments, info = model.transcribe(filepath, beam_size=5, vad_filter=True)
    # 语言信息（第一段输出即答案）
    lang = lang_name(info.language)
    prob = info.language_probability

    print(f"\n=== 转写: {os.path.basename(filepath)} ===")
    print(f"   检测语言: {lang} ({info.language})  置信度: {prob:.1%}")
    print(f"   音频时长: {info.duration:.1f}s (VAD后 {info.duration_after_vad:.1f}s)")

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        print(f"   [{seg.start:7.2f}s -> {seg.end:7.2f}s] {text}")
        lines.append((seg.start, seg.end, text))

    elapsed = time.time() - t0
    print(f"   转写耗时: {elapsed:.1f}s")

    stem = os.path.splitext(os.path.basename(filepath))[0]
    srt_name = f"{stem}_{lang}_{prob:.1%}.srt"

    if srt_dir:
        os.makedirs(srt_dir, exist_ok=True)
        srt_path = os.path.join(srt_dir, srt_name)
        if lines:
            # 正常: 写转写文本
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, (st, en, txt) in enumerate(lines, 1):
                    f.write(f"{i}\n{fmt_time(st)} --> {fmt_time(en)}\n{txt}\n\n")
            print(f"   字幕已保存: {srt_path}")
        else:
            # 无语音: 也生成字幕文件，内容标注原因（文件数和视频数对得上）
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(f"1\n{fmt_time(0)} --> {fmt_time(1)}\n未检测到语音\n\n")
            print(f"   ⚠️ 未检测到语音，已生成占位字幕: {srt_path}")

    status = "ok" if lines else "no_speech"
    return lang, info.language, prob, info.duration, elapsed, status


def get_desktop():
    """返回真实桌面路径（兼容 OneDrive 重定向）"""
    home = os.path.expanduser("~")
    for candidate in (
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive", "桌面"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(home, "Desktop")


def main():
    parser = argparse.ArgumentParser(description="faster-whisper 批量转写")
    parser.add_argument("path", nargs="+", help="音频文件或文件夹路径，可传多个(支持拖放)")
    parser.add_argument("--model", default="small", help="模型名或本地模型目录路径")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="auto=自动检测(默认)")
    parser.add_argument("--srt", help="可选: 指定字幕导出目录")
    parser.add_argument("--no-srt", action="store_true", help="不导出字幕")
    args = parser.parse_args()

    files = collect_files(args.path)
    if not files:
        print("没有找到可转写的音视频文件。")
        sys.exit(1)

    # 字幕默认输出到 桌面\转写字幕
    if args.no_srt:
        srt_dir = None
    elif args.srt:
        srt_dir = args.srt
    else:
        srt_dir = os.path.join(get_desktop(), "转写字幕")
    if srt_dir:
        print(f"字幕将保存到: {srt_dir}")

    print(f"找到 {len(files)} 个文件，加载模型 {args.model} ...")
    t0 = time.time()
    model, device_name = load_model(args.model, args.device)
    print(f"模型加载完成 ({device_name})，耗时 {time.time()-t0:.1f}s")

    print(f"\n{'='*60}\n开始批量转写 {len(files)} 个文件...\n{'='*60}")
    summary = []
    for i, fp in enumerate(files, 1):
        print(f"\n----- [{i}/{len(files)}] {fp} -----")
        try:
            summary.append((os.path.basename(fp), *transcribe_one(model, fp, srt_dir)))
        except Exception as e:
            print(f"   !! 转写失败: {e}")
            summary.append((os.path.basename(fp), "失败", "-", 0, 0, 0, "failed"))

    # 汇总表
    print(f"\n{'='*60}\n汇总\n{'='*60}")
    print(f"{'文件名':<36}{'语言':<12}{'置信度':<10}{'时长':<10}{'耗时':<8}{'状态'}")
    print("-" * 90)
    for row in summary:
        name, lang, code, prob, dur, el, status = row
        if status == "failed":
            print(f"{name:<36}{'转写失败':<10}")
        else:
            status_txt = "" if status == "ok" else "⚠️ 无语音"
            print(f"{name:<36}{lang+'('+code+')':<14}{prob:<11.1%}{dur:.1f}s{'':<6}{el:.1f}s{'':<4}{status_txt}")
    n_ok = sum(1 for r in summary if r[6] == "ok")
    n_nospeech = sum(1 for r in summary if r[6] == "no_speech")
    n_fail = sum(1 for r in summary if r[6] == "failed")
    print(f"\n全部完成: {len(files)} 个文件 (正常 {n_ok} / 无语音 {n_nospeech} / 失败 {n_fail})，总耗时 {sum(r[5] for r in summary if r[6]!='failed'):.1f}s")


if __name__ == "__main__":
    main()
