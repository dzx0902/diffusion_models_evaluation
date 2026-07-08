param(
  [string]$BenchmarkRoot = "",
  [string]$ModelsRoot = "",
  [int]$HunyuanGpus = 1,
  [switch]$IncludeI2V,
  [switch]$SkipExisting,
  [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

function Resolve-ScriptRoot {
  if ($PSScriptRoot) {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
  }
  return (Resolve-Path ".").Path
}

function Write-Step {
  param([string]$Message)
  Write-Host ""
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-OutputFile {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }
  return ((Get-Item -LiteralPath $Path).Length -gt 0)
}

function Assert-PathExists {
  param(
    [string]$Path,
    [string]$Description
  )
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "$Description does not exist: $Path"
  }
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

function Invoke-NativeChecked {
  param(
    [scriptblock]$Body,
    [string]$FailureMessage
  )
  & $Body
  if ($LASTEXITCODE -ne 0) {
    throw "$FailureMessage exited with code $LASTEXITCODE"
  }
}

function Invoke-SmokeTest {
  param(
    [string]$Name,
    [string]$OutputPath,
    [scriptblock]$Body
  )

  Write-Step $Name
  if ($SkipExisting -and (Test-OutputFile -Path $OutputPath)) {
    Write-Host "Skip existing output: $OutputPath"
    return [pscustomobject]@{
      Name = $Name
      Status = "skipped_existing"
      Output = $OutputPath
      Error = ""
    }
  }

  try {
    & $Body
    if (-not (Test-OutputFile -Path $OutputPath)) {
      throw "Smoke output was not created or is empty: $OutputPath"
    }
    Write-Host "OK: $OutputPath" -ForegroundColor Green
    return [pscustomobject]@{
      Name = $Name
      Status = "success"
      Output = $OutputPath
      Error = ""
    }
  }
  catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    if (-not $ContinueOnError) {
      throw
    }
    return [pscustomobject]@{
      Name = $Name
      Status = "failed"
      Output = $OutputPath
      Error = $_.Exception.Message
    }
  }
}

if (-not $BenchmarkRoot) {
  $BenchmarkRoot = Resolve-ScriptRoot
}
$BenchmarkRoot = (Resolve-Path $BenchmarkRoot).Path

if (-not $ModelsRoot) {
  $ModelsRoot = Join-Path $BenchmarkRoot ".ms_video_models"
}
$ModelsRoot = (Resolve-Path $ModelsRoot).Path

$env:MS_BENCHMARK_ROOT = $BenchmarkRoot
$env:MS_MODELS_ROOT = $ModelsRoot
$env:MS_HUNYUAN_GPUS = [string]$HunyuanGpus
$env:USE_LIBUV = "0"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host "Benchmark root: $BenchmarkRoot"
Write-Host "Models root:    $ModelsRoot"
Write-Host "Hunyuan GPUs:   $HunyuanGpus"
Write-Host "USE_LIBUV:      $env:USE_LIBUV"
Write-Host "Include I2V:    $IncludeI2V"
Write-Host "Skip existing:  $SkipExisting"

$results = @()

$hunyuanRepo = Join-Path $ModelsRoot "HunyuanVideo-1.5"
$wanRepo = Join-Path $ModelsRoot "Wan2.2"
$cogRepo = Join-Path $ModelsRoot "CogVideo"
$contentvRepo = Join-Path $ModelsRoot "ContentV"

$results += Invoke-SmokeTest `
  -Name "HunyuanVideo-1.5 T2V smoke test" `
  -OutputPath (Join-Path $hunyuanRepo "outputs\smoke_hunyuan.mp4") `
  -Body {
    Assert-PathExists -Path $hunyuanRepo -Description "HunyuanVideo repo"
    Assert-PathExists -Path (Join-Path $hunyuanRepo "ckpts") -Description "HunyuanVideo ckpts"
    Set-Location $hunyuanRepo
    New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
    Invoke-NativeChecked -FailureMessage "HunyuanVideo T2V smoke test" -Body {
      conda run -n hunyuanvideo15 torchrun --nproc_per_node $env:MS_HUNYUAN_GPUS --rdzv-conf use_libuv=0 generate.py `
        --prompt "A realistic dog walks on an outdoor street." `
        --image_path none `
        --resolution 480p `
        --aspect_ratio 16:9 `
        --seed 0 `
        --rewrite false `
        --cfg_distilled false `
        --sr false `
        --output_path ".\outputs\smoke_hunyuan.mp4" `
        --model_path ".\ckpts"
    }
  }

if ($IncludeI2V) {
  $results += Invoke-SmokeTest `
    -Name "HunyuanVideo-1.5 I2V smoke test" `
    -OutputPath (Join-Path $hunyuanRepo "outputs\smoke_hunyuan_i2v.mp4") `
    -Body {
      Assert-PathExists -Path $hunyuanRepo -Description "HunyuanVideo repo"
      Assert-PathExists -Path (Join-Path $hunyuanRepo "ckpts\vision_encoder\siglip") -Description "HunyuanVideo I2V vision encoder"
      $image = Ensure-TestImage
      Set-Location $hunyuanRepo
      New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
      Invoke-NativeChecked -FailureMessage "HunyuanVideo I2V smoke test" -Body {
        conda run -n hunyuanvideo15 torchrun --nproc_per_node $env:MS_HUNYUAN_GPUS --rdzv-conf use_libuv=0 generate.py `
          --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
          --image_path $image `
          --resolution 480p `
          --aspect_ratio 16:9 `
          --seed 0 `
          --rewrite false `
          --cfg_distilled false `
          --sr false `
          --output_path ".\outputs\smoke_hunyuan_i2v.mp4" `
          --model_path ".\ckpts"
      }
    }
}

$results += Invoke-SmokeTest `
  -Name "Wan2.2-TI2V-5B T2V smoke test" `
  -OutputPath (Join-Path $wanRepo "outputs\smoke_wan_t2v.mp4") `
  -Body {
    Assert-PathExists -Path $wanRepo -Description "Wan2.2 repo"
    Assert-PathExists -Path (Join-Path $wanRepo "Wan2.2-TI2V-5B") -Description "Wan2.2-TI2V-5B weights"
    Set-Location $wanRepo
    New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
    Invoke-NativeChecked -FailureMessage "Wan2.2 T2V smoke test" -Body {
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
  }

if ($IncludeI2V) {
  $results += Invoke-SmokeTest `
    -Name "Wan2.2-TI2V-5B I2V smoke test" `
    -OutputPath (Join-Path $wanRepo "outputs\smoke_wan_i2v.mp4") `
    -Body {
      Assert-PathExists -Path $wanRepo -Description "Wan2.2 repo"
      Assert-PathExists -Path (Join-Path $wanRepo "Wan2.2-TI2V-5B") -Description "Wan2.2-TI2V-5B weights"
      $image = Ensure-TestImage
      Set-Location $wanRepo
      New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
      Invoke-NativeChecked -FailureMessage "Wan2.2 I2V smoke test" -Body {
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

$results += Invoke-SmokeTest `
  -Name "CogVideoX1.5-5B T2V smoke test" `
  -OutputPath (Join-Path $cogRepo "outputs\smoke_cog_t2v.mp4") `
  -Body {
    Assert-PathExists -Path $cogRepo -Description "CogVideo repo"
    Assert-PathExists -Path (Join-Path $ModelsRoot "CogVideoX1.5-5B") -Description "CogVideoX1.5-5B weights"
    Set-Location $cogRepo
    New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
    Invoke-NativeChecked -FailureMessage "CogVideo T2V smoke test" -Body {
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
  }

if ($IncludeI2V) {
  $results += Invoke-SmokeTest `
    -Name "CogVideoX1.5-5B-I2V smoke test" `
    -OutputPath (Join-Path $cogRepo "outputs\smoke_cog_i2v.mp4") `
    -Body {
      Assert-PathExists -Path $cogRepo -Description "CogVideo repo"
      Assert-PathExists -Path (Join-Path $ModelsRoot "CogVideoX1.5-5B-I2V") -Description "CogVideoX1.5-5B-I2V weights"
      $image = Ensure-TestImage
      Set-Location $cogRepo
      New-Item -ItemType Directory -Force -Path ".\outputs" | Out-Null
      Invoke-NativeChecked -FailureMessage "CogVideo I2V smoke test" -Body {
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

$results += Invoke-SmokeTest `
  -Name "ContentV-8B smoke test" `
  -OutputPath (Join-Path $contentvRepo "outputs\smoke_contentv.mp4") `
  -Body {
    Assert-PathExists -Path $contentvRepo -Description "ContentV repo"
    Assert-PathExists -Path (Join-Path $ModelsRoot "ContentV-8B") -Description "ContentV-8B weights"
    Set-Location $env:MS_BENCHMARK_ROOT
    $adapter = Join-Path $env:MS_BENCHMARK_ROOT "scripts\adapters\contentv_generate.py"
    New-Item -ItemType Directory -Force -Path (Join-Path $contentvRepo "outputs") | Out-Null
    Invoke-NativeChecked -FailureMessage "ContentV smoke test" -Body {
      conda run -n contentv8b python $adapter `
        --repo $contentvRepo `
        --model-id (Join-Path $env:MS_MODELS_ROOT "ContentV-8B") `
        --prompt "A realistic dog walks beside a stationary car on an outdoor street." `
        --output (Join-Path $contentvRepo "outputs\smoke_contentv.mp4") `
        --seed 0 `
        --fps 16
    }
  }

Write-Host ""
Write-Host "Smoke test summary" -ForegroundColor Cyan
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.Status -eq "failed" })
if ($failed.Count -gt 0) {
  throw "$($failed.Count) smoke test(s) failed."
}
