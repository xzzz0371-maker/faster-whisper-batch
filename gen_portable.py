# -*- coding: utf-8 -*-
"""生成便携包启动器与使用说明（ASCII cmd + UTF-8 说明）"""
import os

ROOT = r"C:\zdjj\faster-whisper-便携版"

def w(path, content, encoding):
    with open(path, "w", encoding=encoding, newline="") as f:
        f.write(content)

# ---------- 批量转写.cmd ----------
batch = (
    "@echo off\r\n"
    "rem ============================================\r\n"
    "rem  faster-whisper BATCH transcription (portable)\r\n"
    "rem  Usage 1: drag & drop a folder or file(s) onto this icon\r\n"
    "rem  Usage 2: double-click, type a folder/file path, press Enter\r\n"
    "rem ============================================\r\n"
    "cd /d %~dp0\r\n"
    'if not "%~1"=="" goto :has_args\r\n'
    "echo.\r\n"
    "echo  +----------------------------------------------------+\r\n"
    "echo  #   Batch Transcription Tool (faster-whisper)        #\r\n"
    "echo  #   Type a folder or file path, then press Enter     #\r\n"
    "echo  +----------------------------------------------------+\r\n"
    "echo.\r\n"
    "set /p TARGET=Path: \r\n"
    'if "%TARGET%"=="" exit /b\r\n'
    "goto :run\r\n"
    ":has_args\r\n"
    "set TARGET=%*\r\n"
    ":run\r\n"
    'python\\python.exe scripts\\batch_transcribe.py %TARGET% --model "%~dp0models\\medium" --srt "%~dp0输出字幕"\r\n'
    "echo.\r\n"
    "echo Done. Press any key to close...\r\n"
    "pause >nul\r\n"
)
w(os.path.join(ROOT, "批量转写.cmd"), batch, "gbk")

# ---------- 单个转写.cmd ----------
single = (
    "@echo off\r\n"
    "rem ============================================\r\n"
    "rem  faster-whisper SINGLE transcription (portable)\r\n"
    "rem  Drag & drop an audio/video file onto this icon\r\n"
    "rem  or double-click and type the file path\r\n"
    "rem ============================================\r\n"
    "cd /d %~dp0\r\n"
    'if not "%~1"=="" goto :has_args\r\n'
    "echo.\r\n"
    "echo  +----------------------------------------------------+\r\n"
    "echo  #   Transcription Tool (faster-whisper)              #\r\n"
    "echo  #   Type a file path, then press Enter               #\r\n"
    "echo  +----------------------------------------------------+\r\n"
    "echo.\r\n"
    "set /p TARGET=Path: \r\n"
    'if "%TARGET%"=="" exit /b\r\n'
    "goto :run\r\n"
    ":has_args\r\n"
    "set TARGET=%*\r\n"
    ":run\r\n"
    'python\\python.exe scripts\\fw-demo.py %TARGET% "%~dp0models\\medium"\r\n'
    "echo.\r\n"
    "echo Done. Press any key to close...\r\n"
    "pause >nul\r\n"
)
w(os.path.join(ROOT, "单个转写.cmd"), single, "gbk")

# ---------- 使用说明.txt ----------
readme = (
    "faster-whisper 离线转写工具（绿色便携版）\n"
    "=====================================\n"
    "\n"
    "【这是什么】\n"
    "  一个自带运行环境的语音转文字工具。解压后无需安装 Python、\n"
    "  无需联网下载模型，双击即用。\n"
    "\n"
    "【怎么用】\n"
    "  方式一（推荐）：把要转写的【文件夹】或【视频/音频文件】\n"
    "      直接拖到  \"批量转写.cmd\"  图标上，松手即自动转写。\n"
    "      拖多个文件/文件夹也可以。\n"
    "  方式二：双击 \"批量转写.cmd\"，按提示输入文件夹路径后回车。\n"
    "  方式三：转写单个文件，用 \"单个转写.cmd\"（同样支持拖放）。\n"
    "\n"
    "【结果在哪】\n"
    "  转写结果字幕自动保存到本文件夹下的 \"输出字幕\" 目录，\n"
    "  文件名为：原文件名_语种_置信度.srt\n"
    "  例如：画面 (73)_他加禄语_99.4%.srt\n"
    "  SRT 为标准字幕格式，可直接导入剪映 / PR / B站。\n"
    "\n"
    "【支持的格式】\n"
    "  mp3 / wav / flac / m4a / aac / ogg / opus / wma\n"
    "  mp4 / mkv / mov / avi / webm / flv / ts / wmv（视频自动抽音轨）\n"
    "\n"
    "【速度说明】\n"
    "  有 NVIDIA 显卡：自动用 GPU 加速，很快。\n"
    "  没有显卡：自动用 CPU（int8 量化），10 分钟音频约需 1~2 分钟。\n"
    "  1 分钟以内的短视频每条只需几秒。\n"
    "\n"
    "【常见问题】\n"
    "  - 双击后窗口一闪而过？可能被杀毒软件拦截，添加信任即可。\n"
    "  - 转出的文字有错字？音频清晰度不够或口音较重，属正常现象。\n"
    "  - 想更准？联系制作者把模型换成 medium（约1.5GB，解压包会更大）。\n"
    "  - 系统要求：Windows 10/11 64位，内存 4GB 以上。\n"
    "\n"
    "【目录结构】\n"
    "  批量转写.cmd / 单个转写.cmd   入口脚本（双击或拖放）\n"
    "  python\\                       内置运行环境（勿删）\n"
    "  scripts\\                     转写程序（勿删）\n"
    "  models\\medium\\              语音识别模型（勿删）\n"
    "  输出字幕\\                     字幕输出目录\n"
    "\n"
)
w(os.path.join(ROOT, "使用说明.txt"), readme, "utf-8")

print("launchers + readme created")
for n in os.listdir(ROOT):
    print(" ", n)

