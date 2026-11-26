<#
E4 on the CAFNR PC (plan, 2026-11-25), after e1b_e10.ps1. From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e4_ensemble.ps1

No training: it uses the models of e1b_e10.ps1 and e1c_depth.ps1. Scores on test_newdays (Feb 12-15) and
test_newsows (January sows):
  <recipe>_sens   the recipe's three seeds averaged into one model, each also over the left-right mirror of the
                  frame (ir, fusion, depth, pre_ir, pre_fusion; a recipe whose models are missing is skipped)
  fusion_m_s<seed> each fusion seed alone with the mirror only, to see how much of the ensemble's change is the mirror
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

# each recipe's three seeds (fusion: seeds 1-3 of one recipe once e1c_depth.ps1 has trained fusion_s3, else r11 + 1 + 2)
$fusion = @("models\seeds\fusion_s1.pth", "models\seeds\fusion_s2.pth", "models\seeds\fusion_s3.pth")
if (-not (Test-Path $fusion[2])) { $fusion = @($R11, "models\seeds\fusion_s1.pth", "models\seeds\fusion_s2.pth") }
$seeds = [ordered]@{
    "ir"         = @($R10, "models\seeds\ir_s1.pth", "models\seeds\ir_s2.pth")
    "fusion"     = $fusion
    "depth"      = @("models\seeds\depth_s42.pth", "models\seeds\depth_s1.pth", "models\seeds\depth_s2.pth")
    "pre_ir"     = @("models\seeds\pre_ir_s42.pth", "models\seeds\pre_ir_s1.pth", "models\seeds\pre_ir_s2.pth")
    "pre_fusion" = @("models\seeds\pre_fusion_s42.pth", "models\seeds\pre_fusion_s1.pth", "models\seeds\pre_fusion_s2.pth")
}
foreach ($r in @($seeds.Keys)) {
    $missing = @($seeds[$r] | Where-Object { -not (Test-Path $_) })
    if ($missing.Count) { Write-Host "  no ensemble for $r (missing $($missing -join ', '))"; $seeds.Remove($r) }
}
Write-Host ("Ensembles: " + (($seeds.Keys | ForEach-Object { "$_ = " + ($seeds[$_] -join " + ") }) -join "; "))
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
    if ($seeds.Contains("fusion")) {
        foreach ($p in $seeds["fusion"]) {
            $s = 42
            if ($p -match "_s(\d+)\.pth$") { $s = [int]$Matches[1] }
            $dir = "$Out\eval_fusion_m_s$($s)_$($split)"
            if (Test-Path "$dir\eval_report.txt") { continue }
            $a = @("-u", "vulva\eval_seg.py", "--model", $p, "--mirror", "--labels", $Data, "--split", $split, "--out-dir", $dir)
            $evals += @{ Name = "fusion s$s mirror $split"; Log = "$Out\logs\eval_fusion_m_s$($s)_$($split).log"; Args = $a }
        }
    }
}
Write-Host "Scoring: $($evals.Count) evals, 4 at a time ..."
if ($evals.Count) { Invoke-Queue $Py $evals 4 }

Write-Host ""
& $Py vulva\seed_table.py $Out --ref fusion
Write-Host ("`nAll done after {0:N0} min. Paste me everything from the table down (also in $Out\seed_table.txt)." -f ((Get-Date) - $t0).TotalMinutes)
