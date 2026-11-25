<#
E1b + E10 on the CAFNR PC (plan, 2026-11-25). From the repo root (C:\Users\aakdbx\OAK-Camera-Pipeline):

    powershell -ExecutionPolicy Bypass -File vulva\experiments\e1b_e10.ps1

Trains, all on dataset v1's train split (r10's 567 frames), 3 classes, no cross-validation, r10's epochs and zoom:
  ir_s1, ir_s2                          E1b: IR from ImageNet (r10 is ir_s42)
  fusion_s1, fusion_s2                  E1b: IR + height + shape index from ImageNet (r11 is fusion_s42)
  pre_ir_s42, pre_ir_s1, pre_ir_s2      E10: IR started from the Roboflow-pretrained model
  pre_fusion_s42, pre_fusion_s1, _s2    E10 for fusion: IR filters from the Roboflow model, depth filters from zero
then scores all 12 models on test_newdays (Feb 12-15) and test_newsows (January sows) and prints the
seed table (vulva\seed_table.py): mean and spread over seeds, and each recipe against its baseline.

Six trainings share the GPU at a time, as r10 and r11 did (about 1.5 h in all), then the scoring (about 20 min).
Results: outputs\seeds\seed_table.txt, copied to the Mac share (\\tsclient\paal_exchange\seeds).
If it stops (PC needed, disconnect), run it again: finished models and scores are kept and skipped.
-NoFusionPretrain leaves out the three pre_fusion models (about 30 min less).
#>
param(
    [string]$Data = "C:\Users\aakdbx\vulva_dataset_v1\labels.csv",
    [string]$R10 = "models\vulva_seg_r10.pth",
    [string]$R11 = "",
    [string]$Pre = "models\vulva_pre_roboflow.pth",
    [string]$PreShare = "\\tsclient\paal_exchange\vulva_pre_roboflow.pth",
    [string]$Out = "outputs\seeds",
    [string]$CopyTo = "\\tsclient\paal_exchange\seeds",
    [int]$AtOnce = 6,
    [switch]$NoFusionPretrain
)
$ErrorActionPreference = "Stop"
$t0 = Get-Date

# ---------- inputs ----------
if (-not (Test-Path "vulva\train_seg.py")) { throw "Run this from the repo root (the folder that holds vulva\)." }
if (-not (Test-Path $Data)) { throw "Dataset table not found: $Data (pass -Data <labels.csv>)" }
if (-not (Test-Path $R10)) { throw "r10 not found: $R10 (pass -R10 <path>)" }
function Get-Inputs($path) {
    $v = python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('inputs', 'ir'))" $path
    return "$v".Trim()
}
$ErrorActionPreference = "Continue"           # python writes warnings to stderr; failures are checked by hand below
if (-not $R11) {                              # r11 = the fusion model trained on r10's frames
    foreach ($c in (Get-ChildItem models -Filter "vulva_seg_r11*.pth")) {
        if ((Get-Inputs ("models\" + $c.Name)) -eq "fusion") { $R11 = "models\" + $c.Name; break }
    }
    if (-not $R11) { throw "r11 (fusion) not found in models\ (pass -R11 <path>)" }
}
if ((Get-Inputs $R10) -ne "ir") { throw "$R10 is not an IR model (pass -R10 <path>)" }
if (-not (Test-Path $Pre)) {
    if (-not (Test-Path $PreShare)) { throw "Roboflow model not found: $Pre, nor on the share: $PreShare" }
    Write-Host "Copying the Roboflow-pretrained model from the Mac share ..."
    # plain read/write: PowerShell Copy-Item is refused on \\tsclient
    python -c "import shutil, sys; shutil.copyfileobj(open(sys.argv[1], 'rb'), open(sys.argv[2], 'wb'))" $PreShare $Pre
    if (-not (Test-Path $Pre)) { throw "copy failed: $Pre" }
}

function Show-Meta($path) {
    python -c "import sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False); print('  %-45s inputs %-6s classes %s  zoom %s  frames %s' % (sys.argv[1], c.get('inputs', 'ir'), c.get('classes', 2), c.get('scale_aug', ''), c.get('trained_on', '')))" $path
}
Write-Host "Starting points (check: r10 = ir, r11 = fusion, both classes 3):"
Show-Meta $R10; Show-Meta $R11; Show-Meta $Pre

# r10's recipe: epochs and zoom from its report (defaults 60 and 0.9,1.1 if the report is not there)
$Epochs = "60"; $Zoom = "0.9,1.1"
$rep = "outputs\vulva_seg_r10\seg_report.json"
if (Test-Path $rep) {
    $j = Get-Content $rep -Raw | ConvertFrom-Json
    if ($j.epochs) { $Epochs = "$($j.epochs)" }
    if ($j.scale_aug) { $Zoom = "$($j.scale_aug)" }
}
Write-Host "Recipe: r10's settings, epochs $Epochs, zoom $Zoom, 3 classes, train split of $Data"

New-Item -ItemType Directory -Force -Path "$Out\logs", "models\seeds" | Out-Null
$Out = (Resolve-Path $Out).Path

# ---------- a small queue: at most N python processes at once ----------
function Invoke-Queue($items, [int]$n) {
    $running = New-Object System.Collections.ArrayList
    foreach ($it in $items) {
        while (@($running | Where-Object { -not $_.P.HasExited }).Count -ge $n) { Start-Sleep -Seconds 20 }
        $p = Start-Process -FilePath "python" -ArgumentList $it.Args -NoNewWindow -PassThru -WorkingDirectory $PWD.Path `
            -RedirectStandardOutput $it.Log -RedirectStandardError ($it.Log -replace "\.log$", ".err")
        $null = $p.Handle                       # keeps the exit code readable after the process ends
        [void]$running.Add(@{ Name = $it.Name; P = $p; Log = $it.Log })
        Write-Host ("  started {0,-16} {1:HH:mm}" -f $it.Name, (Get-Date))
    }
    while (@($running | Where-Object { -not $_.P.HasExited }).Count -gt 0) { Start-Sleep -Seconds 20 }
    foreach ($r in $running) {
        if ($r.P.ExitCode -ne 0) {
            Write-Host "  FAILED $($r.Name) (exit $($r.P.ExitCode)); last lines of its log:" -ForegroundColor Red
            Get-Content ($r.Log -replace "\.log$", ".err") -Tail 5 -ErrorAction SilentlyContinue
            Get-Content $r.Log -Tail 3 -ErrorAction SilentlyContinue
        }
    }
}

# ---------- training ----------
$plan = @(
    @{ Name = "ir_s1"; Inputs = "ir"; Seed = 1; Init = "" },
    @{ Name = "ir_s2"; Inputs = "ir"; Seed = 2; Init = "" },
    @{ Name = "fusion_s1"; Inputs = "fusion"; Seed = 1; Init = "" },
    @{ Name = "fusion_s2"; Inputs = "fusion"; Seed = 2; Init = "" },
    @{ Name = "pre_ir_s42"; Inputs = "ir"; Seed = 42; Init = $Pre },
    @{ Name = "pre_ir_s1"; Inputs = "ir"; Seed = 1; Init = $Pre },
    @{ Name = "pre_ir_s2"; Inputs = "ir"; Seed = 2; Init = $Pre }
)
if (-not $NoFusionPretrain) {
    $plan += @(
        @{ Name = "pre_fusion_s42"; Inputs = "fusion"; Seed = 42; Init = $Pre },
        @{ Name = "pre_fusion_s1"; Inputs = "fusion"; Seed = 1; Init = $Pre },
        @{ Name = "pre_fusion_s2"; Inputs = "fusion"; Seed = 2; Init = $Pre }
    )
}
$train = @()
foreach ($m in $plan) {
    if (Test-Path "models\seeds\$($m.Name).pth") { Write-Host "  done already: $($m.Name)"; continue }
    $a = @("-u", "vulva\train_seg.py", "--labels", $Data, "--split", "train", "--classes", "3",
           "--inputs", $m.Inputs, "--folds", "0", "--seed", "$($m.Seed)", "--epochs", $Epochs, "--scale-aug", $Zoom,
           "--out-model", "models\seeds\$($m.Name).pth", "--out-dir", "$Out\train_$($m.Name)")
    if ($m.Init) { $a += @("--init-from", $m.Init) }
    $train += @{ Name = $m.Name; Args = $a; Log = "$Out\logs\train_$($m.Name).log" }
}
Write-Host "`nTraining $($train.Count) models, $AtOnce at a time (logs in $Out\logs) ..."
if ($train.Count) { Invoke-Queue $train $AtOnce }
Write-Host ("Training finished after {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
foreach ($m in $plan) {                       # what each model was trained on, and whether it was saved
    $log = "$Out\logs\train_$($m.Name).log"
    if (-not (Test-Path $log)) { continue }
    $s = Select-String -Path $log -Pattern "frames from \d+ sows|tails: looked at|started from|Saved " | ForEach-Object { $_.Line.Trim() }
    Write-Host ("  {0,-16} {1}" -f $m.Name, ($s -join " | "))
}

# ---------- scoring ----------
$models = [ordered]@{ "ir_s42" = $R10; "fusion_s42" = $R11 }
foreach ($m in $plan) { $models[$m.Name] = "models\seeds\$($m.Name).pth" }
$evals = @()
foreach ($name in $models.Keys) {
    if (-not (Test-Path $models[$name])) { Write-Host "  no model for $name, not scored"; continue }
    foreach ($split in @("test_newdays", "test_newsows")) {
        $dir = "$Out\eval_$($name)_$($split)"
        if (Test-Path "$dir\eval_report.txt") { continue }
        $evals += @{ Name = "$($name) $($split)"; Log = "$Out\logs\eval_$($name)_$($split).log";
                     Args = @("-u", "vulva\eval_seg.py", "--model", $models[$name], "--labels", $Data, "--split", $split,
                              "--out-dir", $dir) }
    }
}
Write-Host "`nScoring: $($evals.Count) evals, 4 at a time ..."
if ($evals.Count) { Invoke-Queue $evals 4 }

# ---------- table ----------
Write-Host ""
python vulva\seed_table.py $Out --ref ir --copy-to $CopyTo
Write-Host ("`nAll done after {0:N0} min. Paste me the table above (also in $Out\seed_table.txt)." -f ((Get-Date) - $t0).TotalMinutes)
