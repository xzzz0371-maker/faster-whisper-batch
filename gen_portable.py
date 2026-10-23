# -*- coding: utf-8 -*-
"""生成便携版（绿色包）的目录结构与启动器

用法:
  python gen_portable.py                        # 输出到 ./便携版
  python gen_portable.py D:\\pack               # 指定输出目录
  python gen_portable.py --python-dir <现成python目录> --model <模型目录>

说明:
  - 生成目录结构、复制 scripts/*.py、生成两个 ASCII 启动器和 UTF-8 使用说明
  - .cmd 内容**全部是 ASCII**，路径一律加引号，避免中文代码页导致的乱码目录
    （老版本把中文目录名写进 GBK 的 .cmd 里，非中文 Windows 会把字幕吐进乱码文件夹）
  - --python-dir / --model 可选：把现成的 Python 运行环境和模型复制进包里，
    不传则只生成骨架，并提示你自己放
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "便携版")
SCRIPT_FILES = ("batch_transcribe.py", "fw-demo.py", "whisper_common.py")

LAUNCHER_HEADER = """@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem =====================================================================
rem  faster-whisper {mode} transcription (portable)
rem  Drag a folder / file(s) onto this icon, or double-click and type a path.
rem  This launcher is pure ASCII on purpose: no codepage-dependent text,
rem  and every path is quoted, so spaces and brackets in paths are safe.
rem =====================================================================

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=%~dp0python\\python.exe"
set "SCRIPTS=%~dp0scripts"
set "OUTDIR=%~dp0output"
set "TARGET="
set "ARGS="

if not exist "%PY%" goto :no_python
if not exist "%SCRIPTS%\\{script}" goto :no_scripts

rem ---- model: FA_MODEL env wins, then models\\large-v3, then first subfolder ----
rem      e.g.  set FA_MODEL=%~dp0models\\medium   before running this file
if defined FA_MODEL set "MODEL=%FA_MODEL%"
if not defined MODEL if exist "%~dp0models\\large-v3\\" set "MODEL=%~dp0models\\large-v3"
if not defined MODEL for /d %%D in ("%~dp0models\\*") do if not defined MODEL set "MODEL=%%~fD"
if not defined MODEL goto :no_model

echo.
echo  Model  : %MODEL%
echo  Output : %OUTDIR%
echo.

rem ---- collect arguments, every one quoted ----
:collect
if "%~1"=="" goto :collected
set ARGS=%ARGS% "%~1"
shift
goto :collect

:collected
if defined ARGS goto :run
echo  Type a folder or file path, then press Enter.
echo.
set /p "TARGET=Path: "
if not defined TARGET goto :no_input
set ARGS="%TARGET%"

:run
"%PY%" "%SCRIPTS%\\{script}" %ARGS% {extra}--model "%MODEL%"{srt}
set "RC=%ERRORLEVEL%"
echo.
echo  Exit code: %RC%
if not "%RC%"=="0" echo  [WARN] Something failed. Scroll up for details.
echo  Done. Press any key to close...
pause >nul
exit /b %RC%

:no_input
echo  Nothing entered, bye.
exit /b 0

:no_python
echo  [ERROR] Embedded Python not found:
echo          %PY%
echo  Put a Python runtime folder there first, or run:
echo          python gen_portable.py --python-dir ^<existing python dir^>
pause
exit /b 1

:no_scripts
echo  [ERROR] Scripts not found under:
echo          %SCRIPTS%
pause
exit /b 1

:no_model
echo  [ERROR] No model folder found under:
echo          %~dp0models
echo  Put a CTranslate2 model folder there, for example large-v3.
pause
exit /b 1
"""

README_TXT = """faster-whisper 离线转写工具（绿色便携版）
=====================================

【这是什么】
  一个自带运行环境的语音转文字工具：解压即用，不用装 Python，
  第一次之外也不需要联网（模型就在包内 models 目录里）。

【怎么用】
  方式一（推荐）：把要转写的【文件夹】或【视频/音频文件】直接拖到
          "批量转写.cmd" 图标上松手，自动开始。可一次拖多个。
  方式二：双击 "批量转写.cmd"，按提示输入路径回车。
  方式三：单个文件用 "单个转写.cmd"（同样支持拖放）。

【结果在哪】
  字幕保存在本文件夹下的 "output" 目录（ASCII 名字，避免乱码），
  文件名：原文件名_语种_置信度.srt，例如 画面(73)_他加禄语_99.4%.srt
  同批出现重名时自动加 _2/_3，不会互相覆盖。
  另外会生成 output\\转写日志.txt，记录每个文件的语言/置信度/耗时/失败原因。

【无语音的文件】
  没有检测到人声也会生成一个【空】字幕文件（0 条），
  这样文件数对得上；导入剪映不会多出任何字幕。

【支持的格式】
  mp3 / wav / flac / m4a / aac / ogg / opus / wma
  mp4 / mkv / mov / avi / webm / flv / ts / wmv（视频自动抽音轨）

【速度】
  有 NVIDIA 显卡并且装好 CUDA 12 运行库：走 GPU，很快。
  没有显卡：自动用 CPU（int8 量化）。
  注意：大模型（large-v3 / turbo）在纯 CPU 上会明显慢，可改用 medium 或 small。
  显存不够时程序会自动把精度从 float16 降到 int8_float16，不会直接崩。

【想换模型】
  启动器默认优先用 models\\large-v3，没有就用 models 下第一个子文件夹。
  临时换别的模型，在命令行里先设环境变量再运行：
      set FA_MODEL=%~dp0models\\medium
      批量转写.cmd "D:\\素材"
  或者直接改 批量转写.cmd 里那行 "if not defined MODEL if exist ..."。

【常见问题】
  - 双击后窗口一闪而过：可能被杀毒软件拦截，加信任。
  - 提示 cublas64_12.dll / cudnn 找不到：缺 CUDA 12 运行库，
    跑 pip install nvidia-cublas-cu12 nvidia-cudnn-cu12，
    或改用 CPU 模式（程序会自动回退，不会崩）。
  - 转出的文字有错字：音频清晰度或口音问题，属正常现象。
  - 想要更准：换成 large-v3 模型目录，放进 models 下即可。
  - 系统要求：Windows 10/11 64 位，内存 4GB 以上（large 建议 8GB+）。

【目录结构】
  批量转写.cmd / 单个转写.cmd   启动器（双击或拖放）
  python\\                       内置运行环境（勿删）
  scripts\\                      转写程序（勿删）
  models\\                       语音识别模型（可放多个，默认用 large-v3 或第一个）
  output\\                       字幕与日志输出目录
"""

MODELS_NOTE = """把 CTranslate2 格式的模型文件夹放到这里，例如：

  models\\large-v3\\model.bin
  models\\large-v3\\config.json
  models\\large-v3\\tokenizer.json
  models\\large-v3\\vocabulary.txt

获取方式（任选）：
  1) 联网运行一次 batch_transcribe.py，让它自动下载到 HuggingFace 缓存，
     再把缓存里的模型文件复制到这里（推荐，最省事）。
  2) 从 https://huggingface.co/Systran/faster-whisper-large-v3 手动下载。

可以放多个模型，启动器优先用 models\\large-v3，没有则用第一个子目录。
"""


def w(path: str, content: str, encoding: str = "utf-8", crlf: bool = False) -> None:
    """写文件。cmd/txt 用 CRLF：.cmd 只认 CRLF，LF 会让 goto/标签解析出错。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if crlf:
        content = content.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(path, "w", encoding=encoding, newline="") as f:
        f.write(content)


def copy_tree_if_exists(src: str, dst: str, label: str) -> bool:
    if not src:
        return False
    src = os.path.abspath(src)
    if not os.path.isdir(src):
        print(f"[跳过] {label} 目录不存在: {src}")
        return False
    print(f"[复制] {label}: {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成 faster-whisper 便携版目录结构与启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python gen_portable.py\n"
               "  python gen_portable.py D:\\pack\n"
               "  python gen_portable.py --python-dir C:\\pyembed --model models\\large-v3\n",
    )
    parser.add_argument("out_dir", nargs="?", default=DEFAULT_OUT,
                        help=f"便携包输出目录，默认 {DEFAULT_OUT}")
    parser.add_argument("--python-dir", help="可选：复制一个现成的 Python 运行环境到 <out>/python")
    parser.add_argument("--model", action="append", default=[],
                        help="可选：复制模型目录到 <out>/models（可多次指定）")
    args = parser.parse_args()

    root = os.path.abspath(args.out_dir)

    # 1. 目录骨架（先建目录，老版本就是漏了这一步才必然 FileNotFoundError）
    for sub in ("scripts", "models", "output"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    print(f"[目录] {root}")

    # 2. 复制脚本
    copied = []
    for name in SCRIPT_FILES:
        src = os.path.join(HERE, name)
        if not os.path.isfile(src):
            print(f"[警告] 缺少脚本: {src}")
            continue
        shutil.copy2(src, os.path.join(root, "scripts", name))
        copied.append(name)
    print(f"[脚本] {', '.join(copied) if copied else '（无）'}")

    # 3. 生成启动器（纯 ASCII + CRLF，GBK/UTF-8 代码页都不影响）
    batch_cmd = LAUNCHER_HEADER.format(
        mode="BATCH", script="batch_transcribe.py",
        extra="", srt=' --srt "%OUTDIR%"',
    )
    w(os.path.join(root, "批量转写.cmd"), batch_cmd, "ascii", crlf=True)

    single_cmd = LAUNCHER_HEADER.format(
        mode="SINGLE", script="fw-demo.py",
        extra="", srt="",
    )
    w(os.path.join(root, "单个转写.cmd"), single_cmd, "ascii", crlf=True)

    # 4. 说明文件
    w(os.path.join(root, "使用说明.txt"), README_TXT, crlf=True)
    w(os.path.join(root, "models", "README-放模型到这里.txt"), MODELS_NOTE, crlf=True)

    # 5. 可选：复制运行环境与模型
    copy_tree_if_exists(args.python_dir, os.path.join(root, "python"), "Python 运行环境")
    for m in args.model:
        if os.path.isdir(m):
            name = os.path.basename(os.path.normpath(m))
            copy_tree_if_exists(m, os.path.join(root, "models", name), f"模型 {name}")
        else:
            print(f"[跳过] 模型目录不存在: {m}")

    # 6. 自检 + 结论（判断目录里实际有没有，而不是只看参数传没传）
    python_ok = os.path.isfile(os.path.join(root, "python", "python.exe"))
    models_dir = os.path.join(root, "models")
    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))]
    model_names = [os.path.basename(n) for n in model_dirs]

    print("\n" + "=" * 60)
    print("生成完成，目录内容：")
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        print(f"  {'[目录]' if os.path.isdir(full) else '[文件]'} {name}")
    print("=" * 60)
    print(f"运行环境: {'已就绪' if python_ok else '缺失'}"
          f"    模型: {', '.join(sorted(model_names)) if model_names else '无'}")
    if not python_ok:
        print("⚠ 还没有 python 目录：启动器会直接报错退出。补上再分发：")
        print("    python gen_portable.py --python-dir <现成的 Python 运行环境目录>")
    if not model_names:
        print("⚠ 还没有模型：把 CTranslate2 模型放进 models\\large-v3（或任意子目录）。")
    print("提示：分发前把整个目录压成 zip 即可，接收方解压双击 .cmd 就能用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
