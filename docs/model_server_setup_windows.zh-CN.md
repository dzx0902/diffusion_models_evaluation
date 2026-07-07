# Windows GPU 服务器模型准备命令（仅 5-9B）

本文档给出 Windows + PowerShell 服务器上的规范化部署方式，用于配合本仓库的 `ms_generate.py` 批量评测视频生成模型。

核心原则：

- 本 benchmark 仓库只负责调度、抽帧、评估和报告。
- 视频生成模型需要提前下载到 Windows GPU 服务器。
- 本文档只准备 5-9B 规模模型：HunyuanVideo-1.5 8.3B、Wan2.2-TI2V-5B、CogVideoX1.5-5B / 5B-I2V、ContentV-8B。
- 不需要下载 Wan 14B、旧版 HunyuanVideo 13B、CogVideoX 更大变体或其他全系列权重。
- 每个模型建议使用独立 conda 环境。
- `configs/ms_eval_models.server.yaml` 已改为 Windows/PowerShell 命令模板。

## 0. 目录约定

手动配置时建议使用相对当前部署目录的路径，不要写死盘符：

```powershell
$env:MS_BENCHMARK_ROOT = (Resolve-Path ".").Path
$env:MS_MODELS_ROOT = Join-Path $env:MS_BENCHMARK_ROOT ".m"
$env:MS_HUNYUAN_GPUS = "1"

New-Item -ItemType Directory -Force -Path $env:MS_MODELS_ROOT
```

其中：

- `MS_MODELS_ROOT`：模型仓库和模型权重目录。
- `MS_BENCHMARK_ROOT`：本 benchmark 仓库目录。
- `MS_HUNYUAN_GPUS`：HunyuanVideo 使用的 GPU 进程数，单卡先设为 `1`。

## 1. 一键配置、下载和 smoke test

本仓库提供了 Windows PowerShell 一键脚本：

```powershell
Set-Location <本 benchmark 仓库目录>
.\scripts\setup_windows_models.ps1 `
  -GitProxy "http://127.0.0.1:8001" `
  -HfEndpoint "https://hf-mirror.com"
```

如果只想先完整下载模型仓库和权重，不创建 conda 环境、不跑 smoke test，使用下载专用脚本：

```powershell
Set-Location <本 benchmark 仓库目录>
.\scripts\download_windows_models.ps1
```

该脚本按当前服务器环境默认使用：

- 模型目录：当前仓库下的 `.ms_video_models`
- HF cache：当前模型目录所在盘根目录下的 `hf_cache_ms`，例如 `T:\hf_cache_ms`
- Git 代理：`http://127.0.0.1:8001`
- HF 镜像：`https://hf-mirror.com`
- 不下载需要 `HF_TOKEN` 的 gated vision encoder

脚本会自动把 `MS_BENCHMARK_ROOT` 设置为脚本所在仓库根目录；如果不传 `-ModelsRoot`，默认把模型放到该仓库目录下的 `.m`。这样部署目录移动到哪个盘，模型目录也跟着当前部署目录走。

如果部署目录本身很深，Windows 上 `hf download --local-dir` 可能因为本地缓存路径过长报 `FileNotFoundError`。脚本不会直接用 `--local-dir` 下载到模型目录，而是先下载到模型根目录下较短的 `.hf_cache`，再复制到目标权重目录。

```powershell
Set-Location <本 benchmark 仓库目录>
.\scripts\setup_windows_models.ps1 `
  -SubstDrive "M:" `
  -GitProxy "http://127.0.0.1:8001" `
  -HfEndpoint "https://hf-mirror.com"
```

代理冲突的处理方式：

- `git clone` 阶段只对当前 `git clone` 命令注入 `-c http.proxy=... -c https.proxy=...`，用于访问 GitHub。
- `hf download` 和 `modelscope download` 阶段会临时清空 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 等代理环境变量，并设置 `HF_ENDPOINT`，用于从 Hugging Face 镜像源下载。
- 每个阶段结束后会恢复原来的代理环境变量，不会永久修改系统环境。

常用选项：

```powershell
# 默认把模型放到当前仓库下的 .m
.\scripts\setup_windows_models.ps1 -GitProxy "http://127.0.0.1:8001"

# 部署路径较深时，用短盘符运行，避免 hf 本地缓存路径过长
.\scripts\setup_windows_models.ps1 -SubstDrive "M:" -GitProxy "http://127.0.0.1:8001"

# 下载 I2V 相关权重并运行 I2V smoke test；Hunyuan I2V vision encoder 需要先设置 HF_TOKEN
.\scripts\setup_windows_models.ps1 -GitProxy "http://127.0.0.1:8001" -IncludeI2V

# 如果确实要把模型放到其他位置，再显式传入 -ModelsRoot
.\scripts\setup_windows_models.ps1 -ModelsRoot ".\models" -GitProxy "http://127.0.0.1:8001"

# 只配置仓库、环境和下载权重，不跑 smoke test
.\scripts\setup_windows_models.ps1 -GitProxy "http://127.0.0.1:8001" -NoSmokeTest

# 已经手动准备好 conda 环境时，跳过环境创建和 pip install
.\scripts\setup_windows_models.ps1 -GitProxy "http://127.0.0.1:8001" -SkipConda
```

脚本默认会依次准备 benchmark 环境、HunyuanVideo-1.5、Wan2.2-TI2V-5B、CogVideoX1.5-5B 和 ContentV-8B，并运行 T2V smoke test。

手动执行下载命令时，建议先定义这个辅助函数。它会让 `hf` 先下载到较短的 cache 目录，再把 snapshot 内容复制到目标权重目录，避开 Windows `--local-dir` 深路径缓存问题：

```powershell
function Copy-HfSnapshot {
  param(
    [Parameter(Mandatory=$true)][string[]]$Snapshot,
    [Parameter(Mandatory=$true)][string]$Destination
  )

  $snapshotPath = $Snapshot | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Last 1
  if (-not $snapshotPath) {
    throw "Cannot locate hf snapshot path: $Snapshot"
  }
  New-Item -ItemType Directory -Force -Path $Destination | Out-Null
  Get-ChildItem -LiteralPath $snapshotPath -Force | Copy-Item -Destination $Destination -Recurse -Force
}
```

## 2. 准备 benchmark 环境

```powershell
Set-Location $env:MS_BENCHMARK_ROOT
conda create -n ms-video-eval python=3.10 -y
conda activate ms-video-eval
pip install -r requirements-ms-eval.txt
pip install -U "huggingface_hub[cli]" modelscope
```

如果服务器访问 Hugging Face 慢，可以设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 3. HunyuanVideo-1.5

官方仓库：

```text
https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
```

安装：

```powershell
Set-Location $env:MS_MODELS_ROOT
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
Set-Location (Join-Path $env:MS_MODELS_ROOT "HunyuanVideo-1.5")

conda create -n hunyuanvideo15 python=3.10 -y
conda activate hunyuanvideo15
pip install -r requirements.txt
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
pip install -U "huggingface_hub[cli]" modelscope
```

下载 8.3B 的 480p 基础权重和文本编码器。不要直接执行不带 `--include` 的 `hf download tencent/HunyuanVideo-1.5`，否则会把 720p、distilled、SR 等多套 transformer 权重一起下载下来。

如果只测 T2V，只需要下载 `transformer/480p_t2v/*`：

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "HunyuanVideo-1.5")
Copy-HfSnapshot `
  -Snapshot (hf download tencent/HunyuanVideo-1.5 `
  --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") `
  --include "config.json" `
  --include "scheduler/*" `
  --include "vae/*" `
  --include "transformer/480p_t2v/*" `
  --max-workers 1) `
  -Destination .\ckpts

Copy-HfSnapshot `
  -Snapshot (hf download Qwen/Qwen2.5-VL-7B-Instruct --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") --max-workers 1) `
  -Destination .\ckpts\text_encoder\llm

Copy-HfSnapshot `
  -Snapshot (hf download google/byt5-small --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") --max-workers 1) `
  -Destination .\ckpts\text_encoder\byt5-small

modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir .\ckpts\text_encoder\Glyph-SDXL-v2
```

如果还要测 Hunyuan I2V，再补下载 480p I2V transformer 和 vision encoder。vision encoder 可能需要申请 gated model 权限，拿到 Hugging Face token 后：

```powershell
Copy-HfSnapshot `
  -Snapshot (hf download tencent/HunyuanVideo-1.5 `
  --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") `
  --include "transformer/480p_i2v/*" `
  --max-workers 1) `
  -Destination .\ckpts

Copy-HfSnapshot `
  -Snapshot (hf download black-forest-labs/FLUX.1-Redux-dev `
  --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") `
  --token $env:HF_TOKEN `
  --max-workers 1) `
  -Destination .\ckpts\vision_encoder\siglip
```

单条 T2V smoke test：

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "HunyuanVideo-1.5")
conda run -n hunyuanvideo15 torchrun --nproc_per_node $env:MS_HUNYUAN_GPUS generate.py `
  --prompt "A realistic dog walks on an outdoor street." `
  --image_path none `
  --resolution 480p `
  --aspect_ratio 16:9 `
  --seed 0 `
  --rewrite false `
  --cfg_distilled false `
  --sr false `
  --output_path .\outputs\smoke_hunyuan.mp4 `
  --model_path .\ckpts
```

benchmark YAML 模板：

- `hunyuanvideo_1_5_480p_t2v`
- `hunyuanvideo_1_5_480p_i2v`

## 4. Wan2.2-TI2V-5B

官方仓库：

```text
https://github.com/Wan-Video/Wan2.2
```

安装：

```powershell
Set-Location $env:MS_MODELS_ROOT
git clone https://github.com/Wan-Video/Wan2.2.git
Set-Location (Join-Path $env:MS_MODELS_ROOT "Wan2.2")

conda create -n wan22 python=3.10 -y
conda activate wan22
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

只下载 TI2V-5B 权重。不要下载 `Wan2.2-T2V-A14B`、`Wan2.2-I2V-A14B` 或 Wan2.2 全系列权重。

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "Wan2.2")
Copy-HfSnapshot `
  -Snapshot (hf download Wan-AI/Wan2.2-TI2V-5B --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") --max-workers 1) `
  -Destination .\Wan2.2-TI2V-5B
```

单条 T2V smoke test：

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "Wan2.2")
conda run -n wan22 python generate.py `
  --task ti2v-5B `
  --size 1280*704 `
  --ckpt_dir .\Wan2.2-TI2V-5B `
  --offload_model True `
  --convert_model_dtype `
  --t5_cpu `
  --base_seed 0 `
  --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
  --save_file .\outputs\smoke_wan_t2v.mp4
```

单条 I2V smoke test 需要先准备一张测试图。benchmark 正式运行时会自动生成 pseudo-reference，不需要手动准备。

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "Wan2.2")
conda run -n wan22 python generate.py `
  --task ti2v-5B `
  --size 1280*704 `
  --ckpt_dir .\Wan2.2-TI2V-5B `
  --offload_model True `
  --convert_model_dtype `
  --t5_cpu `
  --base_seed 0 `
  --image "$env:MS_BENCHMARK_ROOT\outputs\ms_eval\pseudo_refs\example.png" `
  --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
  --save_file .\outputs\smoke_wan_i2v.mp4
```

benchmark YAML 模板：

- `wan2_2_ti2v_5b_t2v`
- `wan2_2_ti2v_5b_i2v`

## 5. CogVideoX1.5-5B / CogVideoX1.5-5B-I2V

官方仓库：

```text
https://github.com/zai-org/CogVideo
```

安装：

```powershell
Set-Location $env:MS_MODELS_ROOT
git clone https://github.com/zai-org/CogVideo.git
Set-Location (Join-Path $env:MS_MODELS_ROOT "CogVideo")

conda create -n cogvideox15 python=3.10 -y
conda activate cogvideox15
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

只预下载 5B T2V 和 5B I2V 权重：

```powershell
Set-Location $env:MS_MODELS_ROOT
Copy-HfSnapshot `
  -Snapshot (hf download zai-org/CogVideoX1.5-5B --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") --max-workers 1) `
  -Destination .\CogVideoX1.5-5B
Copy-HfSnapshot `
  -Snapshot (hf download zai-org/CogVideoX1.5-5B-I2V --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") --max-workers 1) `
  -Destination .\CogVideoX1.5-5B-I2V
```

单条 T2V smoke test：

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "CogVideo")
conda run -n cogvideox15 python inference\cli_demo.py `
  --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
  --model_path "$env:MS_MODELS_ROOT\CogVideoX1.5-5B" `
  --generate_type t2v `
  --output_path .\outputs\smoke_cog_t2v.mp4 `
  --num_frames 81 `
  --fps 16 `
  --seed 0 `
  --dtype bfloat16
```

单条 I2V smoke test：

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "CogVideo")
conda run -n cogvideox15 python inference\cli_demo.py `
  --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
  --image_or_video_path "$env:MS_BENCHMARK_ROOT\outputs\ms_eval\pseudo_refs\example.png" `
  --model_path "$env:MS_MODELS_ROOT\CogVideoX1.5-5B-I2V" `
  --generate_type i2v `
  --output_path .\outputs\smoke_cog_i2v.mp4 `
  --num_frames 49 `
  --fps 16 `
  --seed 0 `
  --dtype float16
```

benchmark YAML 模板：

- `cogvideox1_5_5b`
- `cogvideox1_5_5b_i2v`

## 6. ContentV-8B

官方仓库：

```text
https://github.com/bytedance/ContentV
```

安装：

```powershell
Set-Location $env:MS_MODELS_ROOT
git clone https://github.com/bytedance/ContentV.git
Set-Location (Join-Path $env:MS_MODELS_ROOT "ContentV")

conda create -n contentv8b python=3.10 -y
conda activate contentv8b
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

只预下载 8B 权重：

```powershell
Set-Location $env:MS_MODELS_ROOT
Copy-HfSnapshot `
  -Snapshot (hf download ByteDance/ContentV-8B --cache-dir (Join-Path $env:MS_MODELS_ROOT ".hf_cache") --max-workers 1) `
  -Destination .\ContentV-8B
```

说明：官方 `demo.py` 把 prompt 和输出路径写在脚本里，不适合 benchmark 批量调用。本仓库提供了 adapter：

```text
scripts\adapters\contentv_generate.py
```

单条 smoke test：

```powershell
Set-Location $env:MS_BENCHMARK_ROOT
conda run -n contentv8b python scripts\adapters\contentv_generate.py `
  --repo "$env:MS_MODELS_ROOT\ContentV" `
  --model-id "$env:MS_MODELS_ROOT\ContentV-8B" `
  --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
  --output "$env:MS_MODELS_ROOT\ContentV\outputs\smoke_contentv.mp4" `
  --seed 0 `
  --fps 16
```

benchmark YAML 模板：

- `contentv_8b`

## 7. 使用 Windows YAML 模板运行 benchmark

先 dry-run：

```powershell
Set-Location $env:MS_BENCHMARK_ROOT
conda activate ms-video-eval

python scripts\ms_generate.py `
  --models configs\ms_eval_models.server.yaml `
  --dry-run `
  --limit 2 `
  --seeds 0
```

确认 `outputs\ms_eval\metrics\generation_manifest.jsonl` 里的 `command` 没问题后，把要测试的模型改成：

```yaml
enabled: true
```

运行 T2V：

```powershell
python scripts\ms_generate.py `
  --models configs\ms_eval_models.server.yaml `
  --mode t2v `
  --seeds 0 1 2 `
  --skip-existing
```

运行 I2V：

```powershell
python scripts\ms_generate.py `
  --models configs\ms_eval_models.server.yaml `
  --mode i2v `
  --seeds 0 1 2 `
  --skip-existing
```

运行 Wan TI2V：

```powershell
python scripts\ms_generate.py `
  --models configs\ms_eval_models.server.yaml `
  --mode ti2v `
  --seeds 0 1 2 `
  --skip-existing
```

## 8. 后续评估命令

```powershell
python scripts\ms_extract_frames.py --sample-every 4
python scripts\ms_evaluate.py
python scripts\ms_build_report.py
```

完整串联：

```powershell
python scripts\ms_run_benchmark.py `
  --models configs\ms_eval_models.server.yaml `
  --seeds 0 1 2 `
  --sample-every 4 `
  --skip-existing
```

## 9. 注意事项

- 所有模型命令建议先单独 smoke test，通过后再接入 benchmark。
- `configs/ms_eval_models.server.yaml` 默认全部 `enabled: false`，避免误跑大模型。
- 本文档中的下载命令只覆盖当前 YAML 模板里的 5-9B 模型。不要额外执行各官方 README 里下载全系列或 14B/13B 权重的命令。
- HunyuanVideo-1.5 的 I2V 可能需要 gated vision encoder 权限。
- 如果 `hf download` 提示 `Ignoring --include since filenames have been explicitly set`，说明当前 `hf` CLI 把 `--include` 后面的多个模式误解析成了文件名。每个匹配模式都要单独写一个 `--include`，例如 `--include "config.json" --include "scheduler/*"`。
- 如果 Windows 上出现 `.incomplete` 的 `FileNotFoundError`，不要继续用 `hf download --local-dir` 下载到深层模型目录。改用本文档的 `--cache-dir ...\.hf_cache` 下载，然后把 snapshot 内容复制到目标目录；脚本版已经使用这种方式。
- 如果 `hf` 已经下载完成但报 `'gbk' codec can't encode character '\u2713'`，这是 Windows 控制台编码无法打印 `✓ Downloaded`。脚本版已设置 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1` 并使用 `hf download --quiet` 避免这个问题。
- 若一直卡在 `.lock`，先结束其他 `hf` 进程，再删除对应 `.\ckpts\.cache\huggingface\*.lock` 后重试。
- 如果命令里路径包含空格，建议把部署目录放在无空格路径下；脚本默认的 `.m` 会跟随当前仓库目录。
- Windows PowerShell 的多行续行符是反引号。
- ContentV 使用本仓库 adapter 暴露 prompt/output/seed 参数。
