[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TritonSource = "nvcr.io/nvidia/tritonserver:26.04-py3"
$PrometheusSource = "quay.io/prometheus/prometheus:v2.53.4"
# DaoCloud is used only as a transport mirror because Docker Hub is not
# reachable on some mainland networks. The official Docker Hub ARM64 digest
# below prevents the mirror from silently serving different image content.
$GrafanaSource = "docker.m.daocloud.io/grafana/grafana:11.6.0"
$ExpectedTritonArm64Digest = "sha256:aa5fca20416939f243d587d56557d15f9d2bb81de30c6a4464707225250320fb"
$ExpectedPrometheusArm64Digest = "sha256:d4aee9f05c4d8f294fe01de1a3d3db15ba01fa14c9c3cd19c3c0a2780118cba6"
$ExpectedGrafanaArm64Digest = "sha256:c5bb8837eb70910907d8df3a11d0c489bd9e0f761cb67261e8d7d6af0724ddf3"

$TritonBaseTarget = "registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64"
$TritonRuntimeTarget = "registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime"
$PrometheusTarget = "registry.cn-hangzhou.aliyuncs.com/eapil/prometheus:v2.53.4-arm64"
$GrafanaTarget = "registry.cn-hangzhou.aliyuncs.com/eapil/grafana:11.6.0-arm64"

function Invoke-Docker {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$Capture,
        [ValidateRange(1, 10)]
        [int]$MaxAttempts = 3
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        $output = @()
        if ($Capture) {
            $output = & docker @Arguments
        }
        else {
            & docker @Arguments | ForEach-Object { Write-Host $_ }
        }

        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            if ($Capture) {
                return ($output -join [Environment]::NewLine)
            }
            return
        }

        if ($attempt -eq $MaxAttempts) {
            throw "docker $($Arguments -join ' ') failed with exit code $exitCode after $MaxAttempts attempts"
        }

        $delaySeconds = [Math]::Pow(2, $attempt - 1)
        Write-Warning (
            "docker $($Arguments -join ' ') failed with exit code $exitCode; " +
            "retrying in $delaySeconds second(s) ($attempt/$MaxAttempts)"
        )
        Start-Sleep -Seconds $delaySeconds
    }
}

function Get-RemoteDigest {
    param([Parameter(Mandatory = $true)][string]$Image)

    $inspection = Invoke-Docker -Capture -Arguments @(
        "buildx", "imagetools", "inspect", $Image
    )
    $match = [regex]::Match(
        $inspection,
        "(?m)^Digest:\s+(sha256:[0-9a-f]{64})\s*$"
    )
    if (-not $match.Success) {
        throw "Unable to read remote digest for $Image"
    }
    return $match.Groups[1].Value
}

function Get-RemoteImageInfo {
    param([Parameter(Mandatory = $true)][string]$Image)

    $manifestJson = Invoke-Docker -Capture -Arguments @(
        "buildx", "imagetools", "inspect", "--format", "{{json .Manifest}}", $Image
    )
    $manifest = $manifestJson | ConvertFrom-Json
    $platforms = @()
    $arm64Digest = $null

    $manifestListProperty = $manifest.PSObject.Properties["manifests"]
    if ($null -ne $manifestListProperty) {
        foreach ($entry in $manifestListProperty.Value) {
            if ($null -eq $entry.platform) {
                continue
            }
            $platform = "$($entry.platform.os)/$($entry.platform.architecture)"
            $platforms += $platform
            if ($platform -eq "linux/arm64") {
                if ($null -ne $arm64Digest) {
                    throw "Multiple linux/arm64 manifests found for $Image"
                }
                $arm64Digest = $entry.digest
            }
        }
    }
    else {
        $imageJson = Invoke-Docker -Capture -Arguments @(
            "buildx", "imagetools", "inspect", "--format", "{{json .Image}}", $Image
        )
        $imageConfig = $imageJson | ConvertFrom-Json
        $platform = "$($imageConfig.os)/$($imageConfig.architecture)"
        $platforms = @($platform)
        if ($platform -eq "linux/arm64") {
            $arm64Digest = Get-RemoteDigest -Image $Image
        }
    }

    $configDigest = $null
    if ($null -ne $arm64Digest) {
        $arm64ManifestJson = Invoke-Docker -Capture -Arguments @(
            "buildx", "imagetools", "inspect", "--raw", "$Image@$arm64Digest"
        )
        $arm64Manifest = $arm64ManifestJson | ConvertFrom-Json
        if ($null -eq $arm64Manifest.config -or -not $arm64Manifest.config.digest) {
            throw "Unable to read linux/arm64 config digest for $Image"
        }
        $configDigest = $arm64Manifest.config.digest
    }

    return [pscustomobject]@{
        Image = $Image
        Digest = Get-RemoteDigest -Image $Image
        Arm64Digest = $arm64Digest
        ConfigDigest = $configDigest
        Platforms = $platforms
    }
}

function Assert-ContainsArm64 {
    param([Parameter(Mandatory = $true)][string]$Image)

    $info = Get-RemoteImageInfo -Image $Image
    if ($null -eq $info.Arm64Digest) {
        throw "$Image does not publish a linux/arm64 image"
    }
    Write-Host "$Image -> linux/arm64 $($info.Arm64Digest)"
    return $info
}

function Assert-Arm64Only {
    param([Parameter(Mandatory = $true)][string]$Image)

    $info = Assert-ContainsArm64 -Image $Image
    $runtimePlatforms = @(
        $info.Platforms | Where-Object { $_ -ne "unknown/unknown" }
    )
    if ($runtimePlatforms.Count -ne 1 -or $runtimePlatforms[0] -ne "linux/arm64") {
        throw "$Image contains unexpected runtime platforms: $($runtimePlatforms -join ', ')"
    }
    return $info
}

function Publish-PlatformMirror {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target,
        [string]$ExpectedArm64Digest
    )

    $sourceInfo = Assert-ContainsArm64 -Image $Source
    if (
        $ExpectedArm64Digest -and
        $sourceInfo.Arm64Digest -ne $ExpectedArm64Digest
    ) {
        throw "Unexpected linux/arm64 digest for $Source"
    }
    Invoke-Docker -Arguments @("pull", "--platform", "linux/arm64", $Source)
    Invoke-Docker -Arguments @("tag", $Source, $Target)
    Invoke-Docker -Arguments @("push", $Target)
    $targetInfo = Assert-Arm64Only -Image $Target
    # A registry transfer can recompress a layer and legitimately change its
    # manifest digest. The image config digest still proves that filesystem
    # diff IDs, metadata, entrypoint, and environment are unchanged.
    if ($targetInfo.ConfigDigest -ne $sourceInfo.ConfigDigest) {
        throw "Config digest mismatch while mirroring $Source to $Target"
    }
    return $targetInfo
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    Invoke-Docker -Arguments @("version")
    Invoke-Docker -Arguments @("buildx", "version")

    $published = @()
    $published += Publish-PlatformMirror `
        -Source $TritonSource `
        -Target $TritonBaseTarget `
        -ExpectedArm64Digest $ExpectedTritonArm64Digest
    $published += Publish-PlatformMirror `
        -Source $PrometheusSource `
        -Target $PrometheusTarget `
        -ExpectedArm64Digest $ExpectedPrometheusArm64Digest
    $published += Publish-PlatformMirror `
        -Source $GrafanaSource `
        -Target $GrafanaTarget `
        -ExpectedArm64Digest $ExpectedGrafanaArm64Digest

    Invoke-Docker -Arguments @(
        "buildx", "build",
        "--platform", "linux/arm64",
        "--file", "Dockerfile.triton.jetson",
        "--tag", $TritonRuntimeTarget,
        "--push",
        "."
    )
    $published += Assert-Arm64Only -Image $TritonRuntimeTarget

    $published |
        Select-Object Image, Digest, Arm64Digest, ConfigDigest, @{Name = "Platforms"; Expression = { $_.Platforms -join "," }} |
        Format-Table -AutoSize
}
finally {
    Pop-Location
}
