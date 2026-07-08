param(
  [string]$ModelsRoot = "",
  [string]$GitProxy = "",
  [string]$HfEndpoint = "https://hf-mirror.com",
  [string]$SubstDrive = "",
  [int]$HunyuanGpus = 1,
  [switch]$IncludeI2V,
  [switch]$SkipConda,
  [switch]$SkipDownload,
  [switch]$NoSmokeTest
)

$ErrorActionPreference = "Stop"

function Resolve-BenchmarkRoot {
  $scriptPath = $PSCommandPath
  if (-not $scriptPath) {
    $scriptPath = $MyInvocation.MyCommand.Path
  }
  if ($scriptPath) {
    return (Resolve-Path (Join-Path (Split-Path -Parent $scriptPath) "..")).Path
  }
  return (Resolve-Path ".").Path
}

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Body
  )
  Write-Host ""
  Write-Host "==> $Name" -ForegroundColor Cyan
  & $Body
}

function Invoke-NoProxy {
  param([scriptblock]$Body)

  $envNames = @(
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HF_ENDPOINT",
    "PYTHONIOENCODING",
    "PYTHONUTF8"
  )
  $saved = @()
  foreach ($name in $envNames) {
    $saved += [pscustomobject]@{
      Name = $name
      Value = [Environment]::GetEnvironmentVariable($name, "Process")
    }
  }

  try {
    foreach ($name in $envNames) {
      if ($name -ne "HF_ENDPOINT") {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
      }
    }
    if ($HfEndpoint) {
      $env:HF_ENDPOINT = $HfEndpoint
    }
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    & $Body
  }
  finally {
    foreach ($entry in $saved) {
      [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value, "Process")
    }
  }
}

function Invoke-GitClone {
  param(
    [string]$Url,
    [string]$Target
  )

  if (Test-Path -LiteralPath $Target) {
    Write-Host "Skip existing repo: $Target"
    return
  }

  $parent = Split-Path -Parent $Target
  New-Item -ItemType Directory -Force -Path $parent | Out-Null

  if ($GitProxy) {
    git -c "http.proxy=$GitProxy" -c "https.proxy=$GitProxy" clone $Url $Target
  }
  else {
    git clone $Url $Target
  }
  if ($LASTEXITCODE -ne 0) {
    throw "git clone failed for $Url. Check GitProxy: $GitProxy"
  }
}

function New-HfLocalCacheDirs {
  param(
    [string]$LocalDir,
    [string[]]$Patterns = @()
  )

  $downloadRoot = Join-Path $LocalDir ".cache\huggingface\download"
  New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
  foreach ($pattern in $Patterns) {
    $normalized = $pattern.Replace("/", "\")
    $wildcardIndex = $normalized.IndexOf("*")
    if ($wildcardIndex -ge 0) {
      $normalized = $normalized.Substring(0, $wildcardIndex)
    }
    $relativeDir = Split-Path -Parent $normalized
    if ($relativeDir) {
      New-Item -ItemType Directory -Force -Path (Join-Path $downloadRoot $relativeDir) | Out-Null
    }
  }
}

function Resolve-HfSnapshotPath {
  param(
    [object[]]$Output,
    [string]$RepoId
  )

  $candidates = @()
  foreach ($line in $Output) {
    if (-not $line) {
      continue
    }

    $text = [string]$line
    if ($text -match '^\s*path:\s*(.+?)\s*$') {
      $candidates += $Matches[1]
      continue
    }

    try {
      if (Test-Path -LiteralPath $text) {
        $candidates += $text
      }
    }
    catch {
      continue
    }
  }

  $snapshot = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Last 1
  if (-not $snapshot) {
    throw "Cannot locate hf snapshot path for $RepoId. Output: $Output"
  }
  return $snapshot
}

function Invoke-HfSnapshotDownload {
  param(
    [string]$RepoId,
    [string]$LocalDir,
    [string[]]$Include = @(),
    [string]$Token = ""
  )

  $cacheRoot = Join-Path $env:MS_MODELS_ROOT ".hf_cache"
  New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
  New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

  $args = @("download", $RepoId, "--cache-dir", $cacheRoot, "--max-workers", "1")
  foreach ($pattern in $Include) {
    $args += @("--include", $pattern)
  }
  if ($Token) {
    $args += @("--token", $Token)
  }

  $output = & hf @args
  if ($LASTEXITCODE -ne 0) {
    throw "hf download failed for $RepoId"
  }

  $snapshotPath = Resolve-HfSnapshotPath -Output $output -RepoId $RepoId

  Get-ChildItem -LiteralPath $snapshotPath -Force | Copy-Item -Destination $LocalDir -Recurse -Force
}

function Invoke-CondaCreate {
  param(
    [string]$Name,
    [string]$Python = "3.10"
  )

  if ($SkipConda) {
    Write-Host "Skip conda env: $Name"
    return
  }

  $envs = conda env list
  if ($envs -match "^\s*$([regex]::Escape($Name))\s") {
    Write-Host "Conda env exists: $Name"
    return
  }
  conda create -n $Name "python=$Python" -y
}

function Invoke-PipInstall {
  param(
    [string]$EnvName,
    [string[]]$PipArgs
  )

  if ($SkipConda) {
    return
  }
  conda run -n $EnvName pip install @PipArgs
}

function Ensure-TestImage {
  $dir = Join-Path $env:MS_BENCHMARK_ROOT "outputs\ms_eval\pseudo_refs"
  $path = Join-Path $dir "example.png"
  if (Test-Path -LiteralPath $path) {
    return $path
  }

  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Add-Type -AssemblyName System.Drawing
  $bitmap = New-Object System.Drawing.Bitmap 768, 432
  $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
  try {
    $graphics.Clear([System.Drawing.Color]::FromArgb(230, 230, 230))
    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(65, 90, 140))
    $graphics.FillRectangle($brush, 230, 160, 300, 120)
    $brush.Dispose()
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(30, 30, 30)), 6
    $graphics.DrawEllipse($pen, 250, 260, 70, 70)
    $graphics.DrawEllipse($pen, 445, 260, 70, 70)
    $pen.Dispose()
    $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
  }
  finally {
    $graphics.Dispose()
    $bitmap.Dispose()
  }
  return $path
}

function Invoke-HunyuanGenerate {
  param(
    [string]$Prompt,
    [string]$ImagePath,
    [string]$OutputPath
  )

  if ([int]$env:MS_HUNYUAN_GPUS -le 1) {
    conda run -n hunyuanvideo15 python generate.py `
      --prompt $Prompt `
      --image_path $ImagePath `
      --resolution 480p `
      --aspect_ratio "16:9" `
      --seed 0 `
      --rewrite false `
      --cfg_distilled false `
      --sr false `
      --output_path $OutputPath `
      --model_path ".\ckpts"
  }
  else {
    conda run -n hunyuanvideo15 torchrun --nproc_per_node $env:MS_HUNYUAN_GPUS --rdzv-conf use_libuv=false generate.py `
      --prompt $Prompt `
      --image_path $ImagePath `
      --resolution 480p `
      --aspect_ratio "16:9" `
      --seed 0 `
      --rewrite false `
      --cfg_distilled false `
      --sr false `
      --output_path $OutputPath `
      --model_path ".\ckpts"
  }
}

function Setup-BenchmarkEnv {
  Invoke-Step "Benchmark environment" {
    Set-Location $env:MS_BENCHMARK_ROOT
    Invoke-CondaCreate -Name "ms-video-eval"
    Invoke-PipInstall -EnvName "ms-video-eval" -PipArgs @("-r", "requirements-ms-eval.txt")
    Invoke-PipInstall -EnvName "ms-video-eval" -PipArgs @("-U", "huggingface_hub[cli]", "modelscope")
  }
}

function Setup-Hunyuan {
  $repo = Join-Path $env:MS_MODELS_ROOT "HunyuanVideo-1.5"
  Invoke-Step "HunyuanVideo-1.5 repo and env" {
    Invoke-GitClone -Url "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git" -Target $repo
    Set-Location $repo
    Invoke-CondaCreate -Name "hunyuanvideo15"
    Invoke-PipInstall -EnvName "hunyuanvideo15" -PipArgs @("-r", "requirements.txt")
    Invoke-PipInstall -EnvName "hunyuanvideo15" -PipArgs @("-i", "https://mirrors.tencent.com/pypi/simple/", "--upgrade", "tencentcloud-sdk-python")
    Invoke-PipInstall -EnvName "hunyuanvideo15" -PipArgs @("-U", "huggingface_hub[cli]", "modelscope")
  }

  if (-not $SkipDownload) {
    Invoke-Step "HunyuanVideo-1.5 480p weights without proxy" {
      Set-Location $repo
      Invoke-NoProxy {
        Invoke-HfSnapshotDownload -RepoId "tencent/HunyuanVideo-1.5" -LocalDir ".\ckpts" -Include @("config.json", "scheduler/*", "vae/*", "transformer/480p_t2v/*")
        Invoke-HfSnapshotDownload -RepoId "Qwen/Qwen2.5-VL-7B-Instruct" -LocalDir ".\ckpts\text_encoder\llm"
        Invoke-HfSnapshotDownload -RepoId "google/byt5-small" -LocalDir ".\ckpts\text_encoder\byt5-small"
        modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ".\ckpts\text_encoder\Glyph-SDXL-v2"
        if ($IncludeI2V) {
          Invoke-HfSnapshotDownload -RepoId "tencent/HunyuanVideo-1.5" -LocalDir ".\ckpts" -Include @("transformer/480p_i2v/*")
          if ($env:HF_TOKEN) {
            Invoke-HfSnapshotDownload -RepoId "black-forest-labs/FLUX.1-Redux-dev" -LocalDir ".\ckpts\vision_encoder\siglip" -Token $env:HF_TOKEN
          }
          else {
            Write-Warning "Skip Hunyuan I2V vision encoder: HF_TOKEN is not set."
          }
        }
      }
    }
  }

  if (-not $NoSmokeTest) {
    Invoke-Step "HunyuanVideo-1.5 T2V smoke test" {
      Set-Location $repo
      New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
      Invoke-HunyuanGenerate `
        -Prompt "A realistic dog walks on an outdoor street." `
        -ImagePath "none" `
        -OutputPath ".\outputs\smoke_hunyuan.mp4"
    }

    if ($IncludeI2V) {
      Invoke-Step "HunyuanVideo-1.5 I2V smoke test" {
        Set-Location $repo
        $visionEncoder = Join-Path $repo "ckpts\vision_encoder\siglip"
        if (-not (Test-Path -LiteralPath $visionEncoder)) {
          Write-Warning "Skip Hunyuan I2V smoke test: vision encoder is missing at $visionEncoder"
          return
        }
        $image = Ensure-TestImage
        Invoke-HunyuanGenerate `
          -Prompt "A realistic dog walks beside a stationary car on an outdoor street." `
          -ImagePath $image `
          -OutputPath ".\outputs\smoke_hunyuan_i2v.mp4"
      }
    }
  }
}

function Setup-Wan {
  $repo = Join-Path $env:MS_MODELS_ROOT "Wan2.2"
  Invoke-Step "Wan2.2 repo and env" {
    Invoke-GitClone -Url "https://github.com/Wan-Video/Wan2.2.git" -Target $repo
    Set-Location $repo
    Invoke-CondaCreate -Name "wan22"
    Invoke-PipInstall -EnvName "wan22" -PipArgs @("-r", "requirements.txt")
    Invoke-PipInstall -EnvName "wan22" -PipArgs @("-U", "huggingface_hub[cli]")
  }

  if (-not $SkipDownload) {
    Invoke-Step "Wan2.2-TI2V-5B weights without proxy" {
      Set-Location $repo
      Invoke-NoProxy {
        Invoke-HfSnapshotDownload -RepoId "Wan-AI/Wan2.2-TI2V-5B" -LocalDir ".\Wan2.2-TI2V-5B"
      }
    }
  }

  if (-not $NoSmokeTest) {
    Invoke-Step "Wan2.2-TI2V-5B T2V smoke test" {
      Set-Location $repo
      New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
      conda run -n wan22 python generate.py `
        --task ti2v-5B `
        --size "1280*704" `
        --ckpt_dir ".\Wan2.2-TI2V-5B" `
        --offload_model True `
        --convert_model_dtype `
        --t5_cpu `
        --base_seed 0 `
        --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
        --save_file ".\outputs\smoke_wan_t2v.mp4"
    }

    if ($IncludeI2V) {
      Invoke-Step "Wan2.2-TI2V-5B I2V smoke test" {
        Set-Location $repo
        $image = Ensure-TestImage
        conda run -n wan22 python generate.py `
          --task ti2v-5B `
          --size "1280*704" `
          --ckpt_dir ".\Wan2.2-TI2V-5B" `
          --offload_model True `
          --convert_model_dtype `
          --t5_cpu `
          --base_seed 0 `
          --image $image `
          --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
          --save_file ".\outputs\smoke_wan_i2v.mp4"
      }
    }
  }
}

function Setup-CogVideo {
  $repo = Join-Path $env:MS_MODELS_ROOT "CogVideo"
  Invoke-Step "CogVideo repo and env" {
    Invoke-GitClone -Url "https://github.com/zai-org/CogVideo.git" -Target $repo
    Set-Location $repo
    Invoke-CondaCreate -Name "cogvideox15"
    Invoke-PipInstall -EnvName "cogvideox15" -PipArgs @("-r", "requirements.txt")
    Invoke-PipInstall -EnvName "cogvideox15" -PipArgs @("-U", "huggingface_hub[cli]")
  }

  if (-not $SkipDownload) {
    Invoke-Step "CogVideoX1.5-5B weights without proxy" {
      Set-Location $env:MS_MODELS_ROOT
      Invoke-NoProxy {
        Invoke-HfSnapshotDownload -RepoId "zai-org/CogVideoX1.5-5B" -LocalDir ".\CogVideoX1.5-5B"
        if ($IncludeI2V) {
          Invoke-HfSnapshotDownload -RepoId "zai-org/CogVideoX1.5-5B-I2V" -LocalDir ".\CogVideoX1.5-5B-I2V"
        }
      }
    }
  }

  if (-not $NoSmokeTest) {
    Invoke-Step "CogVideoX1.5-5B T2V smoke test" {
      Set-Location $repo
      New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
      conda run -n cogvideox15 python "inference\cli_demo.py" `
        --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
        --model_path (Join-Path $env:MS_MODELS_ROOT "CogVideoX1.5-5B") `
        --generate_type t2v `
        --output_path ".\outputs\smoke_cog_t2v.mp4" `
        --num_frames 81 `
        --fps 16 `
        --seed 0 `
        --dtype bfloat16
    }

    if ($IncludeI2V) {
      Invoke-Step "CogVideoX1.5-5B-I2V smoke test" {
        Set-Location $repo
        $image = Ensure-TestImage
        conda run -n cogvideox15 python "inference\cli_demo.py" `
          --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
          --image_or_video_path $image `
          --model_path (Join-Path $env:MS_MODELS_ROOT "CogVideoX1.5-5B-I2V") `
          --generate_type i2v `
          --output_path ".\outputs\smoke_cog_i2v.mp4" `
          --num_frames 49 `
          --fps 16 `
          --seed 0 `
          --dtype float16
      }
    }
  }
}

function Setup-ContentV {
  $repo = Join-Path $env:MS_MODELS_ROOT "ContentV"
  Invoke-Step "ContentV repo and env" {
    Invoke-GitClone -Url "https://github.com/bytedance/ContentV.git" -Target $repo
    Set-Location $repo
    Invoke-CondaCreate -Name "contentv8b"
    Invoke-PipInstall -EnvName "contentv8b" -PipArgs @("-r", "requirements.txt")
    Invoke-PipInstall -EnvName "contentv8b" -PipArgs @("-U", "huggingface_hub[cli]")
  }

  if (-not $SkipDownload) {
    Invoke-Step "ContentV-8B weights without proxy" {
      Set-Location $env:MS_MODELS_ROOT
      Invoke-NoProxy {
        Invoke-HfSnapshotDownload -RepoId "ByteDance/ContentV-8B" -LocalDir ".\ContentV-8B"
      }
    }
  }

  if (-not $NoSmokeTest) {
    Invoke-Step "ContentV-8B smoke test" {
      Set-Location $env:MS_BENCHMARK_ROOT
      $adapter = Join-Path $env:MS_BENCHMARK_ROOT "scripts\adapters\contentv_generate.py"
      New-Item -ItemType Directory -Force -Path (Join-Path $repo "outputs") | Out-Null
      conda run -n contentv8b python $adapter `
        --repo $repo `
        --model-id (Join-Path $env:MS_MODELS_ROOT "ContentV-8B") `
        --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
        --output (Join-Path $repo "outputs\smoke_contentv.mp4") `
        --seed 0 `
        --fps 16
    }
  }
}

$benchmarkRoot = Resolve-BenchmarkRoot
if ($SubstDrive) {
  $driveName = $SubstDrive.TrimEnd("\")
  if ($driveName -notmatch "^[A-Za-z]:$") {
    throw "SubstDrive must look like 'M:'"
  }
  $existing = & subst $driveName 2>$null
  if (-not $existing) {
    & subst $driveName $benchmarkRoot
  }
  $benchmarkRoot = "$driveName\"
}
if (-not $ModelsRoot) {
  $ModelsRoot = Join-Path $benchmarkRoot ".m"
}

$env:MS_BENCHMARK_ROOT = $benchmarkRoot
$env:MS_MODELS_ROOT = (New-Item -ItemType Directory -Force -Path $ModelsRoot).FullName
$env:MS_HUNYUAN_GPUS = [string]$HunyuanGpus

Write-Host "MS_BENCHMARK_ROOT = $env:MS_BENCHMARK_ROOT"
Write-Host "MS_MODELS_ROOT    = $env:MS_MODELS_ROOT"
Write-Host "MS_HUNYUAN_GPUS   = $env:MS_HUNYUAN_GPUS"
if ($GitProxy) {
  Write-Host "Git clone proxy   = $GitProxy"
}
Write-Host "HF endpoint       = $HfEndpoint"
Write-Host "HF/proxy policy   = git clone uses GitProxy; hf/modelscope downloads clear proxy env"

Setup-BenchmarkEnv
Setup-Hunyuan
Setup-Wan
Setup-CogVideo
Setup-ContentV

Write-Host ""
Write-Host "Done." -ForegroundColor Green
