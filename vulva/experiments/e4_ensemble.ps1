<#
E4 on the CAFNR PC (plan, 2026-11-25), after e1b_e10.ps1. From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e4_ensemble.ps1

No training: it uses the 12 models of e1b_e10.ps1. Scores on test_newdays (Feb 12-15) and test_newsows (January sows):
  <recipe>_sens   the recipe's three seeds averaged into one model, each also over the left-right mirror of the
                  frame (ir, fusion, pre_ir, pre_fusion)
  ir_m_s<seed>    each IR seed alone with the mirror only, to see how much of the ensemble's change is the mirror
then prints the seed table again, now with the ensembles next to the seeds, the outline size against the hand
outlines, and per test which visible vulvas every seed finds, some find and none finds.
About 15 minutes. Results: outputs\seeds\seed_table.txt.
#>
param(
    [string]$Data = "C:\Users\aakdbx\vulva_dataset_v1\labels.csv",
    [string]$R10 = "models\vulva_seg_r10.pth",
    [string]$R11 = "models\vulva_seg_r11_fusion.pth",
    [string]$Out = "outputs\seeds",
    [string]$Python = ""
)
$ErrorActionPreference = "Continue"
$t0 = Get-Date
if (-not (Test-Path "vulva\eval_seg.py")) { throw "Run this from the repo root (the folder that holds vulva\)." }
if (-not (Test-Path $Data)) { throw "Dataset table not found: $Data (pass -Data <labels.csv>)" }
. "$PSScriptRoot\common.ps1"
$Py = Find-TorchPython $Python

$seeds = [ordered]@{
    "ir"         = @($R10, "models\seeds\ir_s1.pth", "models\seeds\ir_s2.pth")
    "fusion"     = @($R11, "models\seeds\fusion_s1.pth", "models\seeds\fusion_s2.pth")
    "pre_ir"     = @("models\seeds\pre_ir_s42.pth", "models\seeds\pre_ir_s1.pth", "models\seeds\pre_ir_s2.pth")
    "pre_fusion" = @("models\seeds\pre_fusion_s42.pth", "models\seeds\pre_fusion_s1.pth", "models\seeds\pre_fusion_s2.pth")
}
foreach ($r in $seeds.Keys) {
    foreach ($p in $seeds[$r]) { if (-not (Test-Path $p)) { throw "missing $p (run e1b_e10.ps1 first)" } }
}
New-Item -ItemType Directory -Force -Path "$Out\logs" | Out-Null
$Out = (Resolve-Path $Out).Path

$evals = @()
foreach ($split in @("test_newdays", "test_newsows")) {
    foreach ($r in $seeds.Keys) {
        $dir = "$Out\eval_$($r)_sens_$($split)"
        if (Test-Path "$dir\eval_report.txt") { continue }
        $a = @("-u", "vulva\eval_seg.py", "--model") + $seeds[$r] + @("--mirror", "--labels", $Data, "--split", $split, "--out-dir", $dir)
        $evals += @{ Name = "$r ensemble $split"; Log = "$Out\logs\eval_$($r)_sens_$($split).log"; Args = $a }
    }
    $irSeeds = @(42, 1, 2)
    for ($k = 0; $k -lt 3; $k++) {
        $s = $irSeeds[$k]
        $dir = "$Out\eval_ir_m_s$($s)_$($split)"
        if (Test-Path "$dir\eval_report.txt") { continue }
        $a = @("-u", "vulva\eval_seg.py", "--model", $seeds["ir"][$k], "--mirror", "--labels", $Data, "--split", $split, "--out-dir", $dir)
        $evals += @{ Name = "ir s$s mirror $split"; Log = "$Out\logs\eval_ir_m_s$($s)_$($split).log"; Args = $a }
    }
}
Write-Host "Scoring: $($evals.Count) evals, 4 at a time ..."
if ($evals.Count) { Invoke-Queue $Py $evals 4 }

Write-Host ""
& $Py vulva\seed_table.py $Out --ref ir
Write-Host ("`nAll done after {0:N0} min. Paste me everything from the table down (also in $Out\seed_table.txt)." -f ((Get-Date) - $t0).TotalMinutes)
