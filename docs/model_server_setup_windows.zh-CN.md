# Windows GPU 服务器模型准备命令

本文档给出 Windows + PowerShell 服务器上的规范化部署方式，用于配合本仓库的 `ms_generate.py` 批量评测视频生成模型。

核心原则：

- 本 benchmark 仓库只负责调度、抽帧、评估和报告。
- 视频生成模型需要提前下载到 Windows GPU 服务器。
- 每个模型建议使用独立 conda 环境。
- `configs/ms_eval_models.server.yaml` 已改为 Windows/PowerShell 命令模板。

## 0. 目录约定

建议在 PowerShell 中设置：

```powershell
$env:MS_MODELS_ROOT = "D:\ms_video_models"
$env:MS_BENCHMARK_ROOT = "D:\diffusion_models_evaluation"
$env:MS_HUNYUAN_GPUS = "1"

New-Item -ItemType Directory -Force -Path $env:MS_MODELS_ROOT
```

其中：

- `MS_MODELS_ROOT`：模型仓库和模型权重目录。
- `MS_BENCHMARK_ROOT`：本 benchmark 仓库目录。
- `MS_HUNYUAN_GPUS`：HunyuanVideo 使用的 GPU 进程数，单卡先设为 `1`。

如果你的仓库在当前工作目录，可以这样设置：

```powershell
$env:MS_BENCHMARK_ROOT = (Resolve-Path ".").Path
```

## 1. 准备 benchmark 环境

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

## 2. HunyuanVideo-1.5

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

下载基础权重和文本编码器：

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "HunyuanVideo-1.5")
hf download tencent/HunyuanVideo-1.5 --local-dir .\ckpts
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir .\ckpts\text_encoder\llm
hf download google/byt5-small --local-dir .\ckpts\text_encoder\byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir .\ckpts\text_encoder\Glyph-SDXL-v2
```

I2V 相关的 vision encoder 可能需要申请 gated model 权限。拿到 Hugging Face token 后：

```powershell
hf download black-forest-labs/FLUX.1-Redux-dev `
  --local-dir .\ckpts\vision_encoder\siglip `
  --token $env:HF_TOKEN
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

## 3. Wan2.2-TI2V-5B

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

下载 5B 权重：

```powershell
Set-Location (Join-Path $env:MS_MODELS_ROOT "Wan2.2")
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir .\Wan2.2-TI2V-5B
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

## 4. CogVideoX1.5-5B / CogVideoX1.5-5B-I2V

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

预下载权重：

```powershell
Set-Location $env:MS_MODELS_ROOT
huggingface-cli download zai-org/CogVideoX1.5-5B --local-dir .\CogVideoX1.5-5B
huggingface-cli download zai-org/CogVideoX1.5-5B-I2V --local-dir .\CogVideoX1.5-5B-I2V
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

## 5. ContentV-8B

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

预下载权重：

```powershell
Set-Location $env:MS_MODELS_ROOT
huggingface-cli download ByteDance/ContentV-8B --local-dir .\ContentV-8B
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

## 6. 使用 Windows YAML 模板运行 benchmark

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

## 7. 后续评估命令

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

## 8. 注意事项

- 所有模型命令建议先单独 smoke test，通过后再接入 benchmark。
- `configs/ms_eval_models.server.yaml` 默认全部 `enabled: false`，避免误跑大模型。
- HunyuanVideo-1.5 的 I2V 可能需要 gated vision encoder 权限。
- 如果命令里路径包含空格，建议把 `MS_MODELS_ROOT` 和 `MS_BENCHMARK_ROOT` 放在无空格路径下，例如 `D:\ms_video_models`。
- Windows PowerShell 的多行续行符是反引号。
- ContentV 使用本仓库 adapter 暴露 prompt/output/seed 参数。

