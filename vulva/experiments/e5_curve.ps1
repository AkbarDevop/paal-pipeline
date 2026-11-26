<#
E5 learning curve on the CAFNR PC (plan, 2026-11-25). From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline),
with the recipe that won E1d (its three seeds are the "all data" point):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e5_curve.ps1 -Base fusion_drop -Inputs fusion -IrDropout 0.5 -Seeds 1,2,3
    powershell -ExecutionPolicy Bypass -File vulva\experiments\e5_curve.ps1 -Base depth -Inputs depth -Seeds 42,1,2

Question: would more training pictures help, and is it more pictures of the same sows or pictures of more sows?
Trains the recipe again, same seeds, on
  <Base>_f50   half of every animal's pictures (all animals)
  <Base>_a50   all pictures of half the animals
with twice the epochs, so every model gets the same number of training steps as the full one, then scores both
tests and prints the seed table against <Base>. If halving the data costs a lot, more data will help; if half the
animals costs much more than half the pictures, it is new animals we need, not more pictures of the same ones.
Six trainings at once (about 50 min), then the scoring. Run it again after a stop: finished parts are skipped.
#>
param(
    [Parameter(Mandatory = $true)][string]$Base,
    [Parameter(Mandatory = $true)][string]$Inputs,
    [string]$IrDropout = "0",
    [string]$Seeds = "1,2,3",
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
Write-Host "Recipe: $Base ($Inputs, IR dropout $IrDropout), epochs $Epochs x 2 on half the data, zoom $Zoom, 3 classes, train split of $Data"
New-Item -ItemType Directory -Force -Path "$Out\logs", "models\seeds" | Out-Null
$Out = (Resolve-Path $Out).Path

$plan = @()
foreach ($seed in ($Seeds -split "[, ]+" | Where-Object { $_ })) {
    $plan += @{ Name = "$($Base)_f50_s$seed"; Seed = $seed; Keep = @("--keep-frames", "0.5") }
    $plan += @{ Name = "$($Base)_a50_s$seed"; Seed = $seed; Keep = @("--keep-animals", "0.5") }
}
$Epochs2 = "$([int]$Epochs * 2)"                     # half the data, twice the epochs: same number of steps
$train = @()
foreach ($m in $plan) {
    if (Test-Path "models\seeds\$($m.Name).pth") { Write-Host "  done already: $($m.Name)"; continue }
    $a = @("-u", "vulva\train_seg.py", "--labels", $Data, "--split", "train", "--classes", "3",
           "--inputs", $Inputs, "--folds", "0", "--seed", "$($m.Seed)", "--epochs", $Epochs2, "--scale-aug", $Zoom,
           "--out-model", "models\seeds\$($m.Name).pth", "--out-dir", "$Out\train_$($m.Name)",
           "--ir-dropout", $IrDropout) + $m.Keep
    $train += @{ Name = $m.Name; Args = $a; Log = "$Out\logs\train_$($m.Name).log" }
}
Write-Host "`nTraining $($train.Count) models at once (logs in $Out\logs) ..."
if ($train.Count) { Invoke-Queue $Py $train 6 }
Write-Host ("Training finished after {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
foreach ($m in $plan) {
    $log = "$Out\logs\train_$($m.Name).log"
    if (-not (Test-Path $log)) { continue }
    $s = Select-String -Path $log -Pattern "Learning curve|Saved " | ForEach-Object { $_.Line.Trim() }
    Write-Host ("  {0,-22} {1}" -f $m.Name, ($s -join " | "))
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
& $Py vulva\seed_table.py $Out --ref $Base
Write-Host ("`nDone after {0:N0} min." -f ((Get-Date) - $t0).TotalMinutes)
