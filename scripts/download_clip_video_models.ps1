param(
  [string]$ModelsRoot = "",
  [string]$HfEndpoint = "https://hf-mirror.com",
  [switch]$SkipAnimateDiff,
  [switch]$SkipZeroScope
)

$ErrorActionPreference = "Stop"

function Resolve-BenchmarkRoot {
  if ($PSScriptRoot) {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
  }
  return (Resolve-Path ".").Path
}

function Invoke-ModelDownload {
  param(
    [string]$RepoId,
    [string]$Destination,
    [Parameter(Mandatory = $true)]
    [string[]]$Files
  )

  Write-Host ""
  Write-Host "==> $RepoId" -ForegroundColor Cyan
  New-Item -ItemType Directory -Force -Path $Destination | Out-Null

  foreach ($file in $Files) {
    $relativePath = $file.Replace("/", [IO.Path]::DirectorySeparatorChar)
    $target = Join-Path $Destination $relativePath
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    if ((Test-Path -LiteralPath $target) -and (Get-Item -LiteralPath $target).Length -gt 0) {
      Write-Host "Skip:  $file"
      continue
    }

    $partial = "$target.partial"
    $url = "$($HfEndpoint.TrimEnd('/'))/$RepoId/resolve/main/$file"
    $curlArgs = @(
      "-L", "--fail", "--show-error",
      "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
      "--connect-timeout", "30",
      "--speed-limit", "1024", "--speed-time", "60",
      "--output", $partial
    )
    if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -gt 0) {
      $curlArgs += @("--continue-at", "-")
    }
    $curlArgs += $url

    Write-Host "Fetch: $file"
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Download failed for $RepoId/$file. Partial data remains at $partial."
    }
    if (-not (Test-Path -LiteralPath $partial) -or (Get-Item -LiteralPath $partial).Length -eq 0) {
      throw "Downloaded file is empty: $RepoId/$file"
    }
    Move-Item -LiteralPath $partial -Destination $target -Force
  }
  Write-Host "Ready: $Destination" -ForegroundColor Green
}

$benchmarkRoot = Resolve-BenchmarkRoot
if (-not $ModelsRoot) {
  $ModelsRoot = Join-Path $benchmarkRoot ".ms_video_models"
}
$ModelsRoot = (New-Item -ItemType Directory -Force -Path $ModelsRoot).FullName
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  throw "curl.exe is required for resumable downloads."
}

Write-Host "Models root: $ModelsRoot"
Write-Host "HF endpoint: $HfEndpoint"
Write-Host "Downloader:  curl.exe (resumable explicit file list)"

if (-not $SkipAnimateDiff) {
  Invoke-ModelDownload `
    -RepoId "stable-diffusion-v1-5/stable-diffusion-v1-5" `
    -Destination (Join-Path $ModelsRoot "AnimateDiff\sd-v1-5") `
    -Files @(
      "model_index.json",
      "scheduler/scheduler_config.json",
      "text_encoder/config.json",
      "text_encoder/model.fp16.safetensors",
      "tokenizer/merges.txt",
      "tokenizer/special_tokens_map.json",
      "tokenizer/tokenizer_config.json",
      "tokenizer/vocab.json",
      "unet/config.json",
      "unet/diffusion_pytorch_model.fp16.safetensors",
      "vae/config.json",
      "vae/diffusion_pytorch_model.fp16.safetensors"
    )
  Invoke-ModelDownload `
    -RepoId "guoyww/animatediff-motion-adapter-v1-5-2" `
    -Destination (Join-Path $ModelsRoot "AnimateDiff\motion-adapter-v1-5-2") `
    -Files @("config.json", "diffusion_pytorch_model.fp16.safetensors")
}

if (-not $SkipZeroScope) {
  Invoke-ModelDownload `
    -RepoId "cerspense/zeroscope_v2_576w" `
    -Destination (Join-Path $ModelsRoot "ZeroScope\zeroscope_v2_576w") `
    -Files @(
      "model_index.json",
      "scheduler/scheduler_config.json",
      "text_encoder/config.json",
      "text_encoder/pytorch_model.bin",
      "tokenizer/merges.txt",
      "tokenizer/special_tokens_map.json",
      "tokenizer/tokenizer_config.json",
      "tokenizer/vocab.json",
      "unet/config.json",
      "unet/diffusion_pytorch_model.bin",
      "vae/config.json",
      "vae/diffusion_pytorch_model.bin"
    )
}

Write-Host ""
Write-Host "CLIP video baseline downloads completed." -ForegroundColor Green
