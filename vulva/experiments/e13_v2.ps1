<#
E13 on the CAFNR PC (plan, 2026-11-26): the diverse labelling round. From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e13_v2.ps1

1. Builds dataset v2 (once): every split of dataset v1 as it is (the tests keep exactly the same frames), plus the
   two new training sets labelled on 2026-11-26 (vulva_diverse_febmar: 6 days Feb 18 - Mar 8, day and night;
   vulva_diverse_jan: Jan 24 and 28, not the Test 2 stalls). The leak checks run as for v1.
2. Shows what the v2 training split covers (data_summary.py).
3. Trains the two candidates on it with the usual settings, three seeds each, six at once:
   depth_v2 (depth only, seeds 42, 1, 2) and fusion_drop_v2 (IR + depth with the IR hidden in half the
   training frames, seeds 1, 2, 3).
4. Scores them on Test 1 and Test 2 and prints the table: depth_v2 is compared with depth, fusion_drop_v2 with
   fusion_drop (same recipe, old data). About 1.5 h. Run it again after a stop: finished parts are skipped.
#>
param(
    [string]$V1 = "C:\Users\aakdbx\vulva_dataset_v1\labels.csv",
    [string]$V2Dir = "C:\Users\aakdbx\vulva_dataset_v2",
    [string]$NewSets = "diverse_febmar=C:\Users\aakdbx\vulva_diverse_febmar\labels.csv;diverse_jan=C:\Users\aakdbx\vulva_diverse_jan\labels.csv",
    [string]$Out = "outputs\seeds",
    [string]$Python = ""
)
$ErrorActionPreference = "Continue"
$t0 = Get-Date
if (-not (Test-Path "vulva\train_seg.py")) { throw "Run this from the repo root (the folder that holds vulva\)." }
. "$PSScriptRoot\common.ps1"
$Py = Find-TorchPython $Python
$Data = "$V2Dir\labels.csv"
if (-not (Test-Path $Data)) {
    if (-not (Test-Path $V1)) { throw "Dataset v1 not found: $V1" }
    $a = @("vulva\build_dataset.py", $V2Dir, "--from-dataset", $V1)
    foreach ($kv in ($NewSets -split ";" | Where-Object { $_ })) {
        $name, $csv = $kv -split "=", 2
        if (-not (Test-Path $csv)) { throw "New label set not found: $csv" }
        $a += @("--add", "train", $name, $csv)
    }
    Write-Host "Building dataset v2 in $V2Dir ..."
    & $Py @a
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Data)) { throw "Dataset v2 was not built (see above)." }
}
Write-Host "`nWhat the v2 training split covers:"
& $Py vulva\data_summary.py $Data --split train

$Epochs = "60"; $Zoom = "0.75,1.15"                 # r10's settings, as e1b_e10.ps1 used for every seed
$rep = "outputs\vulva_seg_r10\seg_report.json"
if (Test-Path $rep) {
    $j = Get-Content $rep -Raw | ConvertFrom-Json
    if ($j.epochs) { $Epochs = "$($j.epochs)" }
    if ($j.scale_aug) { $Zoom = "$($j.scale_aug)" }
}
Write-Host "`nRecipe: epochs $Epochs, zoom $Zoom, 3 classes, train split of $Data"
New-Item -ItemType Directory -Force -Path "$Out\logs", "models\seeds" | Out-Null
$Out = (Resolve-Path $Out).Path

$plan = @(
    @{ Name = "depth_v2_s42"; Inputs = "depth"; Seed = 42; Drop = "0" },
    @{ Name = "depth_v2_s1"; Inputs = "depth"; Seed = 1; Drop = "0" },
    @{ Name = "depth_v2_s2"; Inputs = "depth"; Seed = 2; Drop = "0" },
    @{ Name = "fusion_drop_v2_s1"; Inputs = "fusion"; Seed = 1; Drop = "0.5" },
    @{ Name = "fusion_drop_v2_s2"; Inputs = "fusion"; Seed = 2; Drop = "0.5" },
    @{ Name = "fusion_drop_v2_s3"; Inputs = "fusion"; Seed = 3; Drop = "0.5" }
)
$train = @()
foreach ($m in $plan) {
    if (Test-Path "models\seeds\$($m.Name).pth") { Write-Host "  done already: $($m.Name)"; continue }
    $a = @("-u", "vulva\train_seg.py", "--labels", $Data, "--split", "train", "--classes", "3",
           "--inputs", $m.Inputs, "--folds", "0", "--seed", "$($m.Seed)", "--epochs", $Epochs, "--scale-aug", $Zoom,
           "--out-model", "models\seeds\$($m.Name).pth", "--out-dir", "$Out\train_$($m.Name)",
           "--ir-dropout", $m.Drop)
    $train += @{ Name = $m.Name; Args = $a; Log = "$Out\logs\train_$($m.Name).log" }
}
Write-Host "`nTraining $($train.Count) models at once (logs in $Out\logs) ..."
if ($train.Count) { Invoke-Queue $Py $train 6 }
Write-Host ("Training finished after {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
foreach ($m in $plan) {
    $log = "$Out\logs\train_$($m.Name).log"
    if (-not (Test-Path $log)) { continue }
    $s = Select-String -Path $log -Pattern "frames from \d+ sows|Saved " | ForEach-Object { $_.Line.Trim() }
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
