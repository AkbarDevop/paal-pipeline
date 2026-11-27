<#
E14 on the CAFNR PC (plan, 2026-11-27): dataset v3 on one outline rule, then every candidate trained and scored on it.
From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e14_v3.ps1

1. Builds dataset v3 (once): dataset v2 with the outlines reviewed in label_ellipse.py --review (make_review_set.py
   folders review_tests and review_train_old; only reviewed rows change, the rest stay as they were). Prints how many
   were reviewed in each folder and the outline size per label set afterwards.
2. Trains depth_v3 (depth only, seeds 42, 1, 2) and fusion_drop_v3 (IR + depth, IR hidden in half the frames,
   seeds 1, 2, 3) on v3, six at once.
3. Scores, on the v3 tests (the reviewed ruler), the new models and the earlier ones again (depth and fusion_drop
   trained on v1 and on v2), because scores on the old and the new ruler cannot be compared. Results go to their own
   folder (outputs\seeds_v3), so the table holds only v3-ruler scores.
About 2 h. Run it again after a stop: finished parts are skipped. -Rebuild builds v3 again (after more reviewing);
models already trained on the earlier v3 are then out of date, so retrain them (move models\seeds\*_v3_*.pth away).
#>
param(
    [string]$V2 = "C:\Users\aakdbx\vulva_dataset_v2\labels.csv",
    [string]$V3Dir = "C:\Users\aakdbx\vulva_dataset_v3",
    [string]$Reviews = "C:\Users\aakdbx\review_tests\labels.csv;C:\Users\aakdbx\review_train_old\labels.csv",
    [string]$Out = "outputs\seeds_v3",
    [string]$Python = "",
    [switch]$Rebuild
)
$ErrorActionPreference = "Continue"
$t0 = Get-Date
if (-not (Test-Path "vulva\train_seg.py")) { throw "Run this from the repo root (the folder that holds vulva\)." }
. "$PSScriptRoot\common.ps1"
$Py = Find-TorchPython $Python

# ---------- 1. dataset v3 ----------
Write-Host "Reviews so far:"
$rev = @()
foreach ($r in ($Reviews -split ";" | Where-Object { $_ })) {
    if (-not (Test-Path $r)) { Write-Host "  $r : not there (left out)"; continue }
    $rows = @(Import-Csv $r)
    $done = @($rows | Where-Object { $_.review -ne "" })
    $redrawn = @($done | Where-Object { $_.review -match "redrawn" })
    Write-Host ("  {0} : {1} of {2} frames reviewed ({3} redrawn)" -f $r, $done.Count, $rows.Count, $redrawn.Count)
    if ($r -match "review_tests" -and $done.Count -eq 0) {
        Write-Host "  The tests were not reviewed: v3 tests are still on the old outline rule." -ForegroundColor Yellow
    }
    if ($done.Count) { $rev += $r }
}
$Data = "$V3Dir\labels.csv"
if ($Rebuild -or -not (Test-Path $Data)) {
    if (-not (Test-Path $V2)) { throw "Dataset v2 not found: $V2" }
    $a = @("vulva\build_dataset.py", $V3Dir, "--from-dataset", $V2)
    foreach ($r in $rev) { $a += @("--relabel", $r) }
    Write-Host "`nBuilding dataset v3 in $V3Dir ..."
    & $Py @a
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Data)) { throw "Dataset v3 was not built (see above)." }
}
Write-Host "`nDataset v3 (look at the hand outline sizes per label set):"
& $Py vulva\data_summary.py $Data

# ---------- 2. training on v3 ----------
$Epochs = "60"; $Zoom = "0.75,1.15"
$rep = "outputs\vulva_seg_r10\seg_report.json"
if (Test-Path $rep) {
    $j = Get-Content $rep -Raw | ConvertFrom-Json
    if ($j.epochs) { $Epochs = "$($j.epochs)" }
    if ($j.scale_aug) { $Zoom = "$($j.scale_aug)" }
}
New-Item -ItemType Directory -Force -Path "$Out\logs", "models\seeds" | Out-Null
$Out = (Resolve-Path $Out).Path
$plan = @(
    @{ Name = "depth_v3_s42"; Inputs = "depth"; Seed = 42; Drop = "0" },
    @{ Name = "depth_v3_s1"; Inputs = "depth"; Seed = 1; Drop = "0" },
    @{ Name = "depth_v3_s2"; Inputs = "depth"; Seed = 2; Drop = "0" },
    @{ Name = "fusion_drop_v3_s1"; Inputs = "fusion"; Seed = 1; Drop = "0.5" },
    @{ Name = "fusion_drop_v3_s2"; Inputs = "fusion"; Seed = 2; Drop = "0.5" },
    @{ Name = "fusion_drop_v3_s3"; Inputs = "fusion"; Seed = 3; Drop = "0.5" }
)
$train = @()
foreach ($m in $plan) {
    if (Test-Path "models\seeds\$($m.Name).pth") { Write-Host "  done already: $($m.Name)"; continue }
    $a = @("-u", "vulva\train_seg.py", "--labels", $Data, "--split", "train", "--classes", "3",
           "--inputs", $m.Inputs, "--folds", "0", "--seed", "$($m.Seed)", "--epochs", $Epochs, "--scale-aug", $Zoom,
           "--out-model", "models\seeds\$($m.Name).pth", "--out-dir", "$Out\train_$($m.Name)", "--ir-dropout", $m.Drop)
    $train += @{ Name = $m.Name; Args = $a; Log = "$Out\logs\train_$($m.Name).log" }
}
Write-Host "`nTraining $($train.Count) models on v3, 6 at a time (logs in $Out\logs) ..."
if ($train.Count) { Invoke-Queue $Py $train 6 }
Write-Host ("Training finished after {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)

# ---------- 3. scoring everything on the v3 tests ----------
$models = [ordered]@{}
foreach ($s in @("42", "1", "2")) {
    $models["depth_s$s"] = "models\seeds\depth_s$s.pth"
    $models["depth_v2_s$s"] = "models\seeds\depth_v2_s$s.pth"
    $models["depth_v3_s$s"] = "models\seeds\depth_v3_s$s.pth"
}
foreach ($s in @("1", "2", "3")) {
    $models["fusion_drop_s$s"] = "models\seeds\fusion_drop_s$s.pth"
    $models["fusion_drop_v2_s$s"] = "models\seeds\fusion_drop_v2_s$s.pth"
    $models["fusion_drop_v3_s$s"] = "models\seeds\fusion_drop_v3_s$s.pth"
}
$evals = @()
foreach ($name in $models.Keys) {
    if (-not (Test-Path $models[$name])) { Write-Host "  no model $($models[$name]), not scored"; continue }
    foreach ($split in @("test_newdays", "test_newsows")) {
        $dir = "$Out\eval_$($name)_$($split)"
        if (Test-Path "$dir\eval_report.txt") { continue }
        $evals += @{ Name = "$name $split"; Log = "$Out\logs\eval_$($name)_$($split).log";
                     Args = @("-u", "vulva\eval_seg.py", "--model", $models[$name], "--labels", $Data, "--split", $split,
                              "--out-dir", $dir) }
    }
}
Write-Host "`nScoring on the v3 tests: $($evals.Count) evals, 4 at a time ..."
if ($evals.Count) { Invoke-Queue $Py $evals 4 }

Write-Host ""
& $Py vulva\seed_table.py $Out --ref fusion_drop
Write-Host "`nPictures (colour image; hand outline green, model yellow; mistakes first):"
foreach ($n in @("depth_v3_s42", "fusion_drop_v3_s1")) {
    foreach ($split in @("test_newdays", "test_newsows")) { Write-Host "  $Out\eval_$($n)_$($split)\eval_sheet.jpg" }
}
Write-Host ("`nAll done after {0:N0} min. Paste me everything from the table down." -f ((Get-Date) - $t0).TotalMinutes)
