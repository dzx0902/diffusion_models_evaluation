param(
  [string]$BenchmarkRoot = "",
  [string]$ModelsRoot = "",
  [string]$CacheRoot = "",
  [string]$GitProxy = "http://127.0.0.1:8001",
  [string]$HfEndpoint = "https://hf-mirror.com",
  [switch]$IncludeI2V,
  [switch]$SkipRepos,
  [switch]$SkipModelScope
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

function Invoke-NoProxy {
  param([scriptblock]$Body)

  $names = @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "HF_ENDPOINT", "PYTHONIOENCODING", "PYTHONUTF8")
  $saved = @()
  foreach ($name in $names) {
    $saved += [pscustomobject]@{
      Name = $name
      Value = [Environment]::GetEnvironmentVariable($name, "Process")
    }
  }

  try {
    foreach ($name in $names) {
      if ($name -ne "HF_ENDPOINT") {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
      }
    }
    $env:HF_ENDPOINT = $HfEndpoint
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
    [string]$Destination
  )

  if (Test-Path -LiteralPath $Destination) {
    Write-Host "Repo exists: $Destination"
    return
  }

  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
  if ($GitProxy) {
    git -c "http.proxy=$GitProxy" -c "https.proxy=$GitProxy" clone $Url $Destination
  }
  else {
    git clone $Url $Destination
  }
  if ($LASTEXITCODE -ne 0) {
    throw "git clone failed for $Url. Check GitProxy: $GitProxy"
  }
}

function Copy-DirectoryContents {
  param(
    [string]$Source,
    [string]$Destination
  )

  New-Item -ItemType Directory -Force -Path $Destination | Out-Null
  robocopy $Source $Destination /E /NFL /NDL /NJH /NJS /NP | Out-Null
  if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed from $Source to $Destination with exit code $LASTEXITCODE"
  }
  $global:LASTEXITCODE = 0
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
    throw "Cannot locate downloaded snapshot for $RepoId. hf output: $Output"
  }
  return $snapshot
}

function Invoke-HfDownload {
  param(
    [string]$RepoId,
    [string]$Destination,
    [string[]]$Include = @()
  )

  Write-Step "HF download $RepoId"
  New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null

  $args = @("download", $RepoId, "--cache-dir", $CacheRoot, "--max-workers", "1")
  foreach ($pattern in $Include) {
    $args += @("--include", $pattern)
  }

  $output = & hf @args
  if ($LASTEXITCODE -ne 0) {
    throw "hf download failed for $RepoId"
  }

  $snapshot = Resolve-HfSnapshotPath -Output $output -RepoId $RepoId

  Copy-DirectoryContents -Source $snapshot -Destination $Destination
  Write-Host "Copied to $Destination"
}

if (-not $BenchmarkRoot) {
  $BenchmarkRoot = Resolve-ScriptRoot
}
$BenchmarkRoot = (Resolve-Path $BenchmarkRoot).Path

if (-not $ModelsRoot) {
  $ModelsRoot = Join-Path $BenchmarkRoot ".ms_video_models"
}
$ModelsRoot = (New-Item -ItemType Directory -Force -Path $ModelsRoot).FullName

if (-not $CacheRoot) {
  $driveRoot = ([System.IO.Path]::GetPathRoot($ModelsRoot)).TrimEnd("\")
  $CacheRoot = Join-Path $driveRoot "hf_cache_ms"
}
$CacheRoot = (New-Item -ItemType Directory -Force -Path $CacheRoot).FullName

$env:MS_BENCHMARK_ROOT = $BenchmarkRoot
$env:MS_MODELS_ROOT = $ModelsRoot
$env:HF_ENDPOINT = $HfEndpoint

Write-Host "Benchmark root: $BenchmarkRoot"
Write-Host "Models root:    $ModelsRoot"
Write-Host "HF cache root:  $CacheRoot"
Write-Host "HF endpoint:    $HfEndpoint"
Write-Host "Git proxy:      $GitProxy"
Write-Host "Include I2V:    $IncludeI2V"

if (-not $SkipRepos) {
  Write-Step "Clone model repositories"
  Invoke-GitClone -Url "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git" -Destination (Join-Path $ModelsRoot "HunyuanVideo-1.5")
  Invoke-GitClone -Url "https://github.com/Wan-Video/Wan2.2.git" -Destination (Join-Path $ModelsRoot "Wan2.2")
  Invoke-GitClone -Url "https://github.com/zai-org/CogVideo.git" -Destination (Join-Path $ModelsRoot "CogVideo")
  Invoke-GitClone -Url "https://github.com/bytedance/ContentV.git" -Destination (Join-Path $ModelsRoot "ContentV")
}

Invoke-NoProxy {
  $hunyuanRepo = Join-Path $ModelsRoot "HunyuanVideo-1.5"
  Invoke-HfDownload `
    -RepoId "tencent/HunyuanVideo-1.5" `
    -Destination (Join-Path $hunyuanRepo "ckpts") `
    -Include @("config.json", "scheduler/*", "vae/*", "transformer/480p_t2v/*")

  Invoke-HfDownload `
    -RepoId "Qwen/Qwen2.5-VL-7B-Instruct" `
    -Destination (Join-Path $hunyuanRepo "ckpts\text_encoder\llm")

  Invoke-HfDownload `
    -RepoId "google/byt5-small" `
    -Destination (Join-Path $hunyuanRepo "ckpts\text_encoder\byt5-small")

  if (-not $SkipModelScope) {
    Write-Step "ModelScope download Glyph-SDXL-v2"
    modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir (Join-Path $hunyuanRepo "ckpts\text_encoder\Glyph-SDXL-v2")
  }

  if ($IncludeI2V) {
    Invoke-HfDownload `
      -RepoId "tencent/HunyuanVideo-1.5" `
      -Destination (Join-Path $hunyuanRepo "ckpts") `
      -Include @("transformer/480p_i2v/*")

    Write-Warning "Skipping black-forest-labs/FLUX.1-Redux-dev because HF_TOKEN is not available on this server."
  }

  Invoke-HfDownload `
    -RepoId "Wan-AI/Wan2.2-TI2V-5B" `
    -Destination (Join-Path $ModelsRoot "Wan2.2\Wan2.2-TI2V-5B")

  Invoke-HfDownload `
    -RepoId "zai-org/CogVideoX1.5-5B" `
    -Destination (Join-Path $ModelsRoot "CogVideoX1.5-5B")

  if ($IncludeI2V) {
    Invoke-HfDownload `
      -RepoId "zai-org/CogVideoX1.5-5B-I2V" `
      -Destination (Join-Path $ModelsRoot "CogVideoX1.5-5B-I2V")
  }

  Invoke-HfDownload `
    -RepoId "ByteDance/ContentV-8B" `
    -Destination (Join-Path $ModelsRoot "ContentV-8B")
}

Write-Host ""
Write-Host "Download complete." -ForegroundColor Green
