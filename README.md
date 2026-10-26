# 🎙️ Faster-Whisper 批量转写工具

> 本地运行 · 批量转写 · 自动语言检测 · SRT 字幕导出 · 默认 large-v3

一个基于 **faster-whisper** 的本地语音转文字（ASR）批量工具：把整个文件夹的短视频/音频一次性转完，自动识别**每段说的是什么语言（含置信度）**，导出标准 **SRT 字幕**。识别全在本机完成，音频不上传任何第三方。

> ⚠️ 首次运行需要联网下载模型权重（large-v3 约 3.1GB），之后完全离线。

---

## 功能特性

| 能力 | 说明 |
| --- | --- |
| 🔄 批量转写 | 文件夹递归扫描 mp3/wav/flac/m4a/mp4/mkv/mov 等 16 种音视频格式 |
| 🌍 语言自动检测 | Whisper 支持 99 种语言，输出**中文名 + 语言代码 + 置信度**（如 `印尼语 (id) 99.2%`）；内置 76 种中文名，其余显示原始代码 |
| 📝 SRT 字幕导出 | 标准 UTF-8 字幕，直接导入剪映 / PR / B 站 |
| 🏷️ 智能命名 | `原文件名_语种_置信度.srt`，例如 `画面(73)_他加禄语_99.4%.srt`；**同批重名自动加 `_2`/`_3`，不会互相覆盖** |
| 🔇 无音频处理 | 检测不到人声的文件命名为 `原文件名_无音频.srt`，内容是**空字幕（0 条）**，文件数与输入严格对得上，导入剪映也不会莫名多出一行字 |
| 💻 设备自适应 | 有 NVIDIA 显卡自动走 GPU；GPU 不可用（缺 cuBLAS/cuDNN）自动回退 CPU(int8)，不会崩 |
| 🧾 运行日志 | 每批生成 `转写日志.txt`，记录语言/置信度/耗时/失败原因，控制台关了也能查 |
| 🎒 便携打包 | 可生成"解压即用"绿色包，对方无需装 Python |

---

## 安装

需要 **Python 3.9+**（实测 3.13.0）。Windows / macOS / Linux 均可。

```bash
git clone https://github.com/xzzz0371-maker/faster-whisper-batch.git
cd faster-whisper-batch
pip install -r requirements.txt
```

**GPU 加速（可选，仅 NVIDIA）**：ctranslate2 需要 CUDA 12 的 `cublas64_12.dll` / cuDNN 9。

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

不装也能正常用——程序会用一个空特征矩阵做**真实推理探测**，探测失败自动回退 CPU(int8)，并在控制台说明原因。

装好这两个包后，`site-packages/nvidia/*/bin` 并不在 Windows 的 DLL 搜索路径里，ctranslate2 仍然找不到。两种解决办法（便携包已内置，无需处理）：

```python
# 办法一：在 site-packages 下放一个 sitecustomize.py，把 nvidia 的 bin 目录注册进 DLL 搜索路径
#（便携版就是这么做的，见 python\Lib\site-packages\sitecustomize.py）
# 办法二：把那几个 bin 目录加到 PATH，或把 DLL 复制到 python\ 根目录
```

---

## 快速开始

```bash
# 最简：转写整个文件夹，字幕导出到 桌面\转写字幕，模型默认 large-v3
python batch_transcribe.py D:\视频

# 换小模型 + 指定字幕目录
python batch_transcribe.py D:\视频 --model medium --srt D:\字幕

# 用本地模型目录（离线环境）
python batch_transcribe.py D:\素材 --model models\large-v3

# 强制 CPU / 提高并发
python batch_transcribe.py D:\素材 --device cpu --workers 2

# 单文件
python fw-demo.py 音频.mp4
python fw-demo.py 音频.mp4 medium --srt out.srt
```

> 注意：模型必须用 `--model` 传，**不能**写成 `batch_transcribe.py D:\视频 medium`——那会把 `medium` 当成第二个输入路径。

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `path` | 必填 | 文件或文件夹，可传多个（支持拖放） |
| `--model` | `large-v3` | 模型名或本地模型目录路径 |
| `--device` | `auto` | `auto` / `cpu` / `cuda`（`cuda` 为强制，失败会报错并给修复建议） |
| `--srt` | 桌面\转写字幕 | 字幕导出目录 |
| `--no-srt` | — | 不导出字幕（与 `--srt` 互斥） |
| `--workers` | `1` | 并发线程数；GPU 上 2-4 通常更快，**CPU 上无收益**（int8 已多线程吃满，并发只会互相抢核，实测反而更慢） |
| `--quiet` | 关 | 不打印逐句文本 |

---

## 模型怎么选

| 模型 | 体积 | 语言 | 适用场景 |
| --- | --- | --- | --- |
| `tiny` / `base` | 75MB / 145MB | 多语 | 只求快、内容简单，精度较差，不建议生产用 |
| `small` | 460MB | 多语 | 弱机器快速过一遍 |
| `medium` | 1.5GB | 多语 | **CPU 场景的性价比之选**，比 small 明显准 |
| **`large-v3`**（默认） | 3.1GB | 多语 | 最准，8GB 显存可跑；多语种/小语种首选 |
| `large-v3-turbo` | 1.6GB | 多语 | large 级别的质量，速度快数倍；显存紧张时首选 |
| `distil-large-v3` | 1.5GB | **仅英文** | 英文专用，非英文内容不要用 |

**首次运行会联网下载**，控制台会打印体积和镜像提示。国内网络建议先设镜像：

```bash
# CMD
set HF_ENDPOINT=https://hf-mirror.com
# PowerShell
$env:HF_ENDPOINT='https://hf-mirror.com'
```

下载过程没有进度条、会静默等待，属正常现象。已有离线模型的，直接 `--model <模型目录>`。

---

## 性能参考

实测环境：i5-13400F（10C16T）+ RTX 3060 Ti 8GB / Windows 11 / 10.4s 英文音频。

| 配置 | 耗时 | 实时率 |
| --- | --- | --- |
| CPU medium int8 | 9.7s | 1.08x |
| **GPU medium float16** | **0.91s** | **11.4x** |
| GPU large-v3（float16，显存/内存充裕时） | 约 1.5-2s | 约 5-7x（估算） |

- **GPU 与 CPU 差约 10 倍**，批量跑素材必须修好 GPU。
- 模型加载有一次性开销（CPU 约 3s，GPU 约 6-8s），批量跑时被摊薄。
- 显存/内存吃紧时**自动降精度**：`float16 → int8_float16 → int8`，权重大约减半，精度影响很小，不会直接崩。
- large-v3 的 float16 权重约 3.1GB。8GB 显卡如果已被浏览器/剪辑软件占掉一半显存，或主机内存不足（加载要把 3GB 权重读进内存），会加载失败——程序会打印当前可用内存和替代方案。
- `--workers` 在 CPU 上无收益（int8 已吃满核心），GPU 上才值得开。

---

## 工作原理

### 1. 识别引擎：faster-whisper × CTranslate2

- **Whisper** 是 OpenAI 开源的通用语音识别模型，编码器-解码器 Transformer，在 68 万小时多语言弱监督数据上训练，天然支持多语言识别与语种判断。
- **faster-whisper** 用 **CTranslate2** 推理引擎重写：对权重做 int8 / float16 量化，官方称在保持精度的前提下可达 4 倍以上推理加速、显存占用减半。
- 本项目使用 CTranslate2 转换版模型（HuggingFace `Systran/faster-whisper-*`）。

### 2. VAD 静音过滤

转写前用 **Silero VAD** 扫描音频，只保留"有人声"的片段，跳过静音、背景音乐。控制台输出的 `VAD后 xx.x s` 即有效语音时长——**为 0.0s 说明整段没有可识别人声**。

### 3. 语言检测与置信度

Whisper 解码时对"这段属于哪种语言"输出概率分布（覆盖 99 种语言），取最高者作为结果，该概率即置信度。

| 置信度 | 含义 | 建议 |
| --- | --- | --- |
| ≥ 90% | 非常可靠 | 直接采信 |
| 50% ~ 90% | 基本可靠 | 可采信，注意口音/噪声 |
| ≈ 43.7%（≈1/99） | 在 99 种语言里"瞎猜" | **大概率没有清晰人声**，不代表真是英语 |

### 4. 设备探测

`ctranslate2.get_cuda_device_count()` 只看显卡数量，**不代表能用**——真正决定成败的是 cublas/cudnn 能否加载。所以本项目在探测到显卡后，会额外跑一次空特征矩阵推理来验证，失败则回退 CPU，并把缺什么库打印出来。

---

## 便携版（绿色免安装）

`gen_portable.py` 生成目录结构、启动器和说明；可选把现成运行环境与模型一起复制进去：

```bash
# 只生成骨架
python gen_portable.py

# 生成到指定目录，并带上运行环境和模型
python gen_portable.py D:\pack --python-dir C:\pyembed --model models\large-v3
```

生成的包里：

```
批量转写.cmd / 单个转写.cmd   启动器（纯 ASCII + 路径全部加引号 + CRLF）
python\                       内置运行环境（含 CUDA 12 运行库）
scripts\                      转写程序
models\                       模型（优先用 large-v3，否则用第一个子目录）
output\                       字幕与 转写日志.txt
使用说明.txt                  中文说明（UTF-8）
```

换模型不用改文件，设个环境变量即可覆盖启动器的自动选择：

```bat
set FA_MODEL=%~dp0models\medium
批量转写.cmd "D:\素材"
```

启动器刻意**只用英文和 ASCII**：老版本把中文目录名写进 GBK 编码的 `.cmd`，在非中文代码页（英文系统、`chcp 65001`、UTF-8 Beta）下会把字幕吐进一个乱码文件夹。现在也支持路径里带空格、括号、`&`。

> 便携包体积大（large-v3 压缩后约 1.5GB+），不要提交到仓库，`.gitignore` 已排除。

---

## 常见问题

**Q1：5 个视频只出了 3 个字幕？**
不会。每个视频都会生成一个 .srt：有人声的写正常内容，没人声的命名成 `原文件名_无音频.srt`、内容为**空字幕（0 条）**。所以文件数一定和视频数一致，导入剪映/PR 得到 0 条字幕，不会多出文字。

**Q2：为什么有的文件叫 `xxx_无音频.srt`？**
`vad_filter` 判断整段没有有效人声（VAD 后时长 0.0s），通常是纯背景音乐/空镜素材。此时语言检测的"语种 + 置信度"是瞎猜的（常见 nn/no 之类），写进文件名会误导，所以统一叫 `_无音频`。

**Q2：字幕里出现莫名其妙的一行字？**
如果字幕条数明显多于实际说话次数，多半是模型在静音/音乐段"幻听"。换 `large-v3`，或确认 `vad_filter` 生效（默认开启），必要时把音频转成 wav 再处理。

**Q3：置信度很低 / 识别不准？**
换更大的模型（`medium` → `large-v3`）；确认音频里确实有人声、噪声别太大；小语种用 `large-v3` 效果明显优于 `small`。

**Q4：报 `cublas64_12.dll is not found`？**
缺 CUDA 12 运行库。`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`，或加 `--device cpu`。程序默认会自动回退，不会中断任务。

**Q5：字幕是乱码？**
SRT 为 UTF-8 编码，剪映/PR 直接导入即可；记事本打开请选 UTF-8。

**Q6：转写中途 Ctrl+C 了，前面的结果还在吗？**
在。已完成的字幕已经落盘，日志也会记录已完成部分。

**Q7：报 `mkl_malloc: failed to allocate memory` / 显存不足？**
这多半不是显卡问题：
- `mkl_malloc` 是**主机内存**不足——模型权重要先读进内存，large-v3 要约 3GB 连续空间。关掉浏览器/剪辑软件，或改用 `--model medium`。
- `CUDA failed with error out of memory` 是**显存**不足。程序会自动降精度重试（float16 → int8_float16 → int8）；8GB 显卡建议关掉占显存的程序，或直接用 `medium`。
- 加载前程序会打印"可用内存 X GB / 模型约 Y GB"，数值明显不够时别硬上。
- 指定的大模型加载不了时，程序会自动改用**同一 models 目录下更小的模型**继续干活，并明确提示用了哪个——不会让整批任务白跑。

**Q8：想多人/多机批量跑？**
`--workers` 提高单机并发；跨机器直接把文件夹分片传给不同机器，`--model` 指向共享目录里的模型即可。

---

## 项目结构

```
├── batch_transcribe.py   # 批量转写主程序（语言检测 + SRT + 设备自适应 + 日志）
├── fw-demo.py            # 单文件转写
├── whisper_common.py     # 公共模块：语言表、模型加载、GPU 探测、格式化
├── gen_portable.py       # 生成便携版目录结构与启动器
├── requirements.txt      # 依赖
├── LICENSE               # MIT
└── .gitattributes        # 换行符约束（.cmd 必须 CRLF）
```

## 许可证

MIT License，见 [LICENSE](LICENSE)。faster-whisper / CTranslate2 / Silero VAD 亦为 MIT；Whisper 模型权重版权归 OpenAI，使用请遵循其各自条款。
