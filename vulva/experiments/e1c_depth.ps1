<#
E1c on the CAFNR PC (plan, 2026-11-25), after e1b_e10.ps1. From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e1c_depth.ps1

The finder takes depth (Akbar, 2026-11-25): depth only, or IR + depth (fusion). This trains, like e1b_e10.ps1
(dataset v1's train split, r10's settings, 3 classes, no cross-validation):
  depth_s42, depth_s1, depth_s2   depth only (height, shape index), three seeds
  fusion_s3                       a third fusion seed with the same settings as fusion_s1 and fusion_s2
                                  (r11 was trained with zoom 0.9-1.1, the others with 0.75-1.15)
r11's own scores are moved out of the fusion group (shown as "r11"), so fusion = seeds 1, 2, 3 of one recipe.
Scores the new models on test_newdays and test_newsows and prints the seed table. Four trainings share
the GPU (about 45 min), then the scoring (about 5 min). Run it again after a stop: finished parts are skipped.
#>
param(
    [string]$Data = "C:\Users\aakdbx\vulva_dataset_v1\labels.csv",
    [string]$Out = "outputs\seeds",
    [string]$Python = ""
)
$ErrorActionPreference = "Continue"
$t0 = Get-Date
if (-not (Test-Path "vulva\train_seg.py")) { throw "Run this from the repo root (the folder that holds vulva\)." }
if (-not (Test-Path $Data)) { throw "Dataset table not found: $Data (pass -Data <labels.csv>)" }
. "$PSScriptRoot\common.ps1"
$Py = Find-TorchPython $Python

$Epochs = "60"; $Zoom = "0.75,1.15"                 # r10's settings, as e1b_e10.ps1 used for every seed
$rep = "outputs\vulva_seg_r10\seg_report.json"
if (Test-Path $rep) {
    $j = Get-Content $rep -Raw | ConvertFrom-Json
    if ($j.epochs) { $Epochs = "$($j.epochs)" }
    if ($j.scale_aug) { $Zoom = "$($j.scale_aug)" }
}
Write-Host "Recipe: epochs $Epochs, zoom $Zoom, 3 classes, train split of $Data"
New-Item -ItemType Directory -Force -Path "$Out\logs", "models\seeds" | Out-Null
$Out = (Resolve-Path $Out).Path

# r11 (zoom 0.9-1.1) out of the fusion group: its scores are shown as the recipe "r11"
foreach ($split in @("test_newdays", "test_newsows")) {
    $old = "$Out\eval_fusion_s42_$($split)"; $new = "$Out\eval_r11_s42_$($split)"
    if ((Test-Path $old) -and -not (Test-Path $new)) { Rename-Item -Path $old -NewName "eval_r11_s42_$($split)"; Write-Host "  r11's $split scores -> $new" }
}

$plan = @(
    @{ Name = "depth_s42"; Inputs = "depth"; Seed = 42 },
    @{ Name = "depth_s1"; Inputs = "depth"; Seed = 1 },
    @{ Name = "depth_s2"; Inputs = "depth"; Seed = 2 },
    @{ Name = "fusion_s3"; Inputs = "fusion"; Seed = 3 }
)
$train = @()
foreach ($m in $plan) {
    if (Test-Path "models\seeds\$($m.Name).pth") { Write-Host "  done already: $($m.Name)"; continue }
    $a = @("-u", "vulva\train_seg.py", "--labels", $Data, "--split", "train", "--classes", "3",
           "--inputs", $m.Inputs, "--folds", "0", "--seed", "$($m.Seed)", "--epochs", $Epochs, "--scale-aug", $Zoom,
           "--out-model", "models\seeds\$($m.Name).pth", "--out-dir", "$Out\train_$($m.Name)")
    $train += @{ Name = $m.Name; Args = $a; Log = "$Out\logs\train_$($m.Name).log" }
}
Write-Host "`nTraining $($train.Count) models at once (logs in $Out\logs) ..."
if ($train.Count) { Invoke-Queue $Py $train 4 }
Write-Host ("Training finished after {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
foreach ($m in $plan) {
    $log = "$Out\logs\train_$($m.Name).log"
    if (-not (Test-Path $log)) { continue }
    $s = Select-String -Path $log -Pattern "frames from \d+ sows|Saved " | ForEach-Object { $_.Line.Trim() }
    Write-Host ("  {0,-12} {1}" -f $m.Name, ($s -join " | "))
}

$evals = @()
foreach ($m in $plan) {
    $model = "models\seeds\$($m.Name).pth"
    if (-not (Test-Path $model)) { Write-Host "  no model for $($m.Name), not scored"; continue }
    foreach ($split in @("test_newdays", "test_newsows")) {
        $dir = "$Out\eval_$($m.Name)_$($split)"
        if (Test-Path "$dir\eval_report.txt") { continue }
        $evals += @{ Name = "$($m.Name) $split"; Log = "$Out\logs\eval_$($m.Name)_$($split).log";
                     Args = @("-u", "vulva\eval_seg.py", "--model", $model, "--labels", $Data, "--split", $split, "--out-dir", $dir) }
    }
}
Write-Host "`nScoring: $($evals.Count) evals, 4 at a time ..."
if ($evals.Count) { Invoke-Queue $Py $evals 4 }

Write-Host ""
& $Py vulva\seed_table.py $Out --ref fusion
Write-Host ("`nDone after {0:N0} min." -f ((Get-Date) - $t0).TotalMinutes)
