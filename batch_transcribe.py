# -*- coding: utf-8 -*-
"""faster-whisper 批量转写脚本

用法:
  python batch_transcribe.py <文件夹或文件...> [--model 模型名或目录] [--srt 输出目录] [--no-srt]

示例:
  python batch_transcribe.py D:\\我的音频
  python batch_transcribe.py D:\\我的音频 --model medium --srt D:\\字幕
  python batch_transcribe.py D:\\素材 --model models\\large-v3 --device cpu
  python batch_transcribe.py D:\\素材 --workers 2

说明:
  - 遍历文件夹下所有音视频文件(递归)，逐个转写
  - 每个文件自动检测语言，输出中文名+代码+置信度
  - 默认导出同名 .srt 字幕到 桌面\\转写字幕
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Set

# 嵌入式 Python（python313._pth）不会把脚本目录进 sys.path，
# 不手动补一下，便携版里 import whisper_common 会直接 ModuleNotFoundError。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_common import (  # noqa: E402  (必须在 sys.path 补齐之后导入)
    MODEL_SIZES,
    announce_model,
    fmt_time,
    lang_name,
    load_model,
    pad,
    truncate_disp,
    warn_if_slow_on_cpu,
)

DEFAULT_MODEL = "large-v3"

AUDIO_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".wmv",
)

# Windows 单路径上限 260，留点余量给扩展名和系统开销
MAX_PATH_LEN = 250


@dataclass
class Result:
    name: str
    path: str
    lang: str = "-"
    code: str = "-"
    prob: float = 0.0
    duration: float = 0.0
    elapsed: float = 0.0
    status: str = "failed"          # ok / no_speech / failed
    error: str = ""
    srt_path: str = ""


def collect_files(paths: List[str], skip_dirs: List[str]) -> List[str]:
    """收集待转写文件：可传多个文件或文件夹(递归)，自动去重，跳过输出目录"""
    skip = {os.path.normcase(os.path.abspath(p)) for p in skip_dirs if p}
    files, seen = [], set()

    def _skipped(path: str) -> bool:
        return os.path.normcase(os.path.abspath(path)) in skip

    for path in paths:
        if os.path.isfile(path):
            real = os.path.abspath(path)
            if real not in seen:
                seen.add(real)
                files.append(real)
        elif os.path.isdir(path):
            if _skipped(path):
                continue
            for root, dirs, names in os.walk(path):
                # 不让输出目录混进扫描结果
                dirs[:] = [d for d in dirs if not _skipped(os.path.join(root, d))]
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


def build_srt_path(out_dir: str, src_path: str, lang: str, prob: float, used: Set[str]) -> str:
    """生成不重复的字幕路径。

    - 同一批里不同目录存在同名文件时，用父目录名做前缀，仍冲突再加 _2/_3
    - 路径过长时截断文件名，避免 Windows 260 字符限制
    """
    stem = os.path.splitext(os.path.basename(src_path))[0]
    base = f"{stem}_{lang}_{prob:.1%}"

    candidate = base
    if candidate.lower() in used:
        parent = os.path.basename(os.path.dirname(src_path)) or "root"
        candidate = f"{parent}_{base}"

    index = 2
    final = candidate
    while final.lower() in used:
        final = f"{candidate}_{index}"
        index += 1
    used.add(final.lower())

    full = os.path.join(out_dir, final + ".srt")
    if len(full) > MAX_PATH_LEN:
        keep = max(20, len(final) - (len(full) - MAX_PATH_LEN))
        final = final[:keep]
        used.add(final.lower())
        full = os.path.join(out_dir, final + ".srt")
    return full


def write_srt(srt_path: str, lines: List[tuple]) -> None:
    """写 SRT。lines 为空时写空文件：条数为 0，导入剪映/PR 后不会多出字幕。"""
    with open(srt_path, "w", encoding="utf-8", newline="\n") as f:
        for i, (start, end, text) in enumerate(lines, 1):
            f.write(f"{i}\n{fmt_time(start)} --> {fmt_time(end)}\n{text}\n\n")


def transcribe_one(model, filepath: str, srt_dir: Optional[str], used_names: Set[str],
                   quiet: bool = False) -> Result:
    """转写单个文件，返回 Result"""
    res = Result(name=os.path.basename(filepath), path=filepath)
    t0 = time.time()
    segments, info = model.transcribe(filepath, beam_size=5, vad_filter=True)

    res.lang = lang_name(info.language)
    res.code = info.language
    res.prob = info.language_probability
    res.duration = info.duration

    print(f"\n=== 转写: {res.name} ===")
    print(f"   检测语言: {res.lang} ({res.code})  置信度: {res.prob:.1%}")
    print(f"   音频时长: {info.duration:.1f}s (VAD后 {info.duration_after_vad:.1f}s)")

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if not quiet:
            print(f"   [{seg.start:7.2f}s -> {seg.end:7.2f}s] {text}")
        lines.append((seg.start, seg.end, text))

    res.elapsed = time.time() - t0
    print(f"   转写耗时: {res.elapsed:.1f}s")

    res.status = "ok" if lines else "no_speech"

    if srt_dir:
        os.makedirs(srt_dir, exist_ok=True)
        srt_path = build_srt_path(srt_dir, filepath, res.lang, res.prob, used_names)
        write_srt(srt_path, lines)
        res.srt_path = srt_path
        if lines:
            print(f"   字幕已保存: {srt_path}")
        else:
            # 空文件而不是"未检测到语音"占位——占位文本会被剪映当真字幕烧进画面
            print(f"   未检测到语音，已生成空字幕(0 条): {srt_path}")
    return res


def get_desktop() -> str:
    """返回真实桌面路径（兼容 OneDrive 重定向）"""
    home = os.path.expanduser("~")
    for candidate in (
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive", "桌面"),
        os.path.join(home, "桌面"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(home, "Desktop")


def write_log(log_path: str, results: List[Result], device_tag: str, model_arg: str,
              interrupted: bool) -> None:
    """把整批结果落盘，控制台关了也查得到"""
    try:
        with open(log_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {model_arg}    设备: {device_tag}\n")
            f.write(f"文件数: {len(results)}"
                    + ("（被中断，仅记录已完成部分）" if interrupted else "") + "\n")
            f.write("-" * 100 + "\n")
            for r in results:
                f.write(f"[{r.status}] {r.path}\n")
                f.write(f"         语言={r.lang}({r.code}) 置信度={r.prob:.1%} "
                        f"时长={r.duration:.1f}s 耗时={r.elapsed:.1f}s\n")
                if r.srt_path:
                    f.write(f"         字幕={r.srt_path}\n")
                if r.error:
                    f.write(f"         错误={r.error}\n")
        print(f"运行日志: {log_path}")
    except OSError as exc:
        print(f"(日志写入失败: {exc})")


def print_summary(results: List[Result], total_files: int) -> None:
    """汇总表：按显示宽度对齐，中文文件名不再错列"""
    w_name, w_lang, w_prob, w_dur, w_el = 36, 16, 10, 12, 10
    print(f"\n{'='*94}\n汇总\n{'='*94}")
    header = (pad("文件名", w_name) + pad("语言", w_lang) + pad("置信度", w_prob)
              + pad("时长", w_dur) + pad("耗时", w_el) + "状态")
    print(header)
    print("-" * 94)

    for r in results:
        if r.status == "failed":
            print(pad(truncate_disp(r.name, w_name), w_name)
                  + pad("转写失败", w_lang) + f"({r.error[:28]})")
            continue
        status_txt = "无语音" if r.status == "no_speech" else "ok"
        print(pad(truncate_disp(r.name, w_name), w_name)
              + pad(f"{r.lang}({r.code})", w_lang)
              + pad(f"{r.prob:.1%}", w_prob)
              + pad(f"{r.duration:.1f}s", w_dur)
              + pad(f"{r.elapsed:.1f}s", w_el)
              + status_txt)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_no = sum(1 for r in results if r.status == "no_speech")
    n_bad = sum(1 for r in results if r.status == "failed")
    total_el = sum(r.elapsed for r in results)
    print(f"\n共 {total_files} 个文件，已完成 {len(results)} 个"
          f"（正常 {n_ok} / 无语音 {n_no} / 失败 {n_bad}），"
          f"转写累计耗时 {total_el:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="faster-whisper 批量转写（默认模型 large-v3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python batch_transcribe.py D:\\视频\n"
               "  python batch_transcribe.py D:\\视频 --model medium --srt D:\\字幕\n"
               "  python batch_transcribe.py a.mp4 b.mp4 --model models\\large-v3 --device cpu\n",
    )
    parser.add_argument("path", nargs="+", help="音频文件或文件夹路径，可传多个(支持拖放)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"模型名或本地模型目录，默认 {DEFAULT_MODEL}")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto=自动检测(默认)，cuda=强制GPU")
    srt_group = parser.add_mutually_exclusive_group()
    srt_group.add_argument("--srt", help="字幕导出目录，默认 桌面\\转写字幕")
    srt_group.add_argument("--no-srt", action="store_true", help="不导出字幕")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发线程数，默认 1（串行）。GPU 上 2-4 通常更快；"
                             "CPU 上几乎无收益（int8 已多线程吃满 CPU，并发只会互相抢核）")
    parser.add_argument("--quiet", action="store_true", help="不打印逐句文本")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers 必须 >= 1")

    # 防呆：把模型名写在位置参数里（老文档的坑）会静默用错模型，这里直接报错
    misplaced = [p for p in args.path if not os.path.exists(p) and p in MODEL_SIZES]
    if misplaced:
        parser.error(
            f"'{misplaced[0]}' 看起来是模型名，但被当成了输入路径。"
            f"模型请用 --model {misplaced[0]} 传入。"
        )

    # 字幕输出目录：默认 桌面\转写字幕
    if args.no_srt:
        srt_dir = None
        print("已指定 --no-srt，不导出字幕")
    else:
        srt_dir = args.srt or os.path.join(get_desktop(), "转写字幕")
        print(f"字幕将保存到: {srt_dir}")

    files = collect_files(args.path, [srt_dir] if srt_dir else [])
    if not files:
        print("没有找到可转写的音视频文件。")
        print(f"支持的扩展名: {', '.join(AUDIO_EXTS)}")
        return 1

    print(f"找到 {len(files)} 个文件，准备加载模型 {args.model} ...")
    announce_model(args.model)

    t0 = time.time()
    try:
        model, device_tag = load_model(args.model, args.device)
    except Exception as exc:
        print(f"\n模型加载失败: {exc}")
        return 2
    print(f"模型加载完成 ({device_tag})，耗时 {time.time()-t0:.1f}s")
    warn_if_slow_on_cpu(device_tag, args.model)
    if args.workers > 1 and device_tag.startswith("cpu"):
        print("提示: CPU 模式下多线程没有收益（int8 已吃满所有核心），"
              "实测 --workers 2 反而更慢，建议保持默认 1。")

    used_names: Set[str] = set()
    results: List[Result] = []
    total = len(files)
    log_path = os.path.join(srt_dir, "转写日志.txt") if srt_dir else None

    print(f"\n{'='*60}\n开始批量转写 {total} 个文件（并发 {args.workers}）...\n{'='*60}")

    interrupted = False
    try:
        if args.workers == 1:
            for i, fp in enumerate(files, 1):
                print(f"\n----- [{i}/{total}] {fp} -----")
                try:
                    results.append(transcribe_one(model, fp, srt_dir, used_names, args.quiet))
                except Exception as exc:
                    print(f"   !! 转写失败: {exc}")
                    results.append(Result(name=os.path.basename(fp), path=fp, error=str(exc)))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(transcribe_one, model, fp, srt_dir, used_names, True): fp
                    for fp in files
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    fp = futures[fut]
                    try:
                        res = fut.result()
                        results.append(res)
                        print(f"[{i}/{total}] {res.status:9s} {res.name}  "
                              f"{res.lang}({res.code}) {res.prob:.1%} {res.elapsed:.1f}s")
                    except Exception as exc:
                        print(f"[{i}/{total}] failed    {os.path.basename(fp)}  ({exc})")
                        results.append(Result(name=os.path.basename(fp), path=fp, error=str(exc)))
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n已中断（Ctrl+C）。已完成的结果保留，下面只汇总已完成的文件。")

    order = {fp: i for i, fp in enumerate(files)}
    results.sort(key=lambda r: order.get(r.path, 9999))

    print_summary(results, total)
    if log_path:
        os.makedirs(srt_dir, exist_ok=True)
        write_log(log_path, results, device_tag, args.model, interrupted)

    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
