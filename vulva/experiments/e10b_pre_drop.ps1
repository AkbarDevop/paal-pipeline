<#
E10b on the CAFNR PC (plan, 2026-11-25), after e1d_irdrop.ps1. From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e10b_pre_drop.ps1

The finder recipe is now fusion with the IR hidden in half the training frames (fusion_drop). This gives the 2022
Roboflow pretraining a fair try with that recipe: the same training (seeds 1, 2, 3, dataset v1's train split,
r10's settings, 3 classes, --ir-dropout 0.5), but started from models\vulva_pre_roboflow.pth (its IR filters;
depth filters from zero) instead of ImageNet. Scores both tests and prints the table: pre_fusion_drop is
compared with fusion_drop. Three trainings at once (about 30 min), then the scoring.
#>
param(
    [string]$Data = "C:\Users\aakdbx\vulva_dataset_v1\labels.csv",
    [string]$Pre = "models\vulva_pre_roboflow.pth",
    [string]$Out = "outputs\seeds",
    [string]$Python = ""
)
$ErrorActionPreference = "Continue"
$t0 = Get-Date
if (-not (Test-Path "vulva\train_seg.py")) { throw "Run this from the repo root (the folder that holds vulva\)." }
if (-not (Test-Path $Data)) { throw "Dataset table not found: $Data (pass -Data <labels.csv>)" }
. "$PSScriptRoot\common.ps1"
$Py = Find-TorchPython $Python
if (-not (Test-Path $Pre)) { throw "Roboflow-pretrained model not found: $Pre" }

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

$plan = @(
    @{ Name = "pre_fusion_drop_s1"; Inputs = "fusion"; Seed = 1 },
    @{ Name = "pre_fusion_drop_s2"; Inputs = "fusion"; Seed = 2 },
    @{ Name = "pre_fusion_drop_s3"; Inputs = "fusion"; Seed = 3 }
)
$train = @()
foreach ($m in $plan) {
    if (Test-Path "models\seeds\$($m.Name).pth") { Write-Host "  done already: $($m.Name)"; continue }
    $a = @("-u", "vulva\train_seg.py", "--labels", $Data, "--split", "train", "--classes", "3",
           "--inputs", $m.Inputs, "--folds", "0", "--seed", "$($m.Seed)", "--epochs", $Epochs, "--scale-aug", $Zoom,
           "--out-model", "models\seeds\$($m.Name).pth", "--out-dir", "$Out\train_$($m.Name)",
           "--ir-dropout", "0.5", "--init-from", $Pre)
    $train += @{ Name = $m.Name; Args = $a; Log = "$Out\logs\train_$($m.Name).log" }
}
Write-Host "`nTraining $($train.Count) models at once (logs in $Out\logs) ..."
if ($train.Count) { Invoke-Queue $Py $train 3 }
Write-Host ("Training finished after {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
foreach ($m in $plan) {
    $log = "$Out\logs\train_$($m.Name).log"
    if (-not (Test-Path $log)) { continue }
    $s = Select-String -Path $log -Pattern "started from|Saved " | ForEach-Object { $_.Line.Trim() }
    Write-Host ("  {0,-20} {1}" -f $m.Name, ($s -join " | "))
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
& $Py vulva\seed_table.py $Out --ref fusion_drop
Write-Host ("`nDone after {0:N0} min." -f ((Get-Date) - $t0).TotalMinutes)
