# Shared by the experiment scripts (dot-sourced): the Python that has torch with CUDA, and a small queue
# that runs at most N Python processes at once. Same code as in e1b_e10.ps1, which ran on the CAFNR PC.

function Get-TorchInfo($exe) {
    try { $o = & $exe -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>$null }
    catch { return $null }
    if ($LASTEXITCODE -ne 0) { return $null }
    return "$o".Trim()
}

function Find-TorchPython([string]$Python = "") {
    # an active venv or conda env, .venv / venv / env in the repo or in folders under the user folder, then python / py
    $cands = @()
    if ($Python) { $cands += $Python }
    if ($env:VIRTUAL_ENV) { $cands += "$env:VIRTUAL_ENV\Scripts\python.exe" }
    if ($env:CONDA_PREFIX) { $cands += "$env:CONDA_PREFIX\python.exe" }
    foreach ($v in @(".venv", "venv", "env")) { $cands += "$($PWD.Path)\$v\Scripts\python.exe" }
    foreach ($d in (Get-ChildItem $HOME -Directory -ErrorAction SilentlyContinue)) {
        $cands += "$($d.FullName)\Scripts\python.exe"
        foreach ($v in @(".venv", "venv", "env")) { $cands += "$($d.FullName)\$v\Scripts\python.exe" }
        foreach ($e in (Get-ChildItem "$($d.FullName)\envs" -Directory -ErrorAction SilentlyContinue)) { $cands += "$($e.FullName)\python.exe" }
    }
    $cands += @("python", "py")
    foreach ($c in ($cands | Select-Object -Unique)) {
        if ($c -match "\\" -and -not (Test-Path $c)) { continue }
        $info = Get-TorchInfo $c
        if ($info -and $info -match "cuda True") { Write-Host "Python: $c ($info)"; return $c }
    }
    throw "No Python with torch and CUDA found. Activate the venv you train in, or pass -Python <path to its python.exe>."
}

function Invoke-Queue($Py, $items, [int]$n) {
    $running = New-Object System.Collections.ArrayList
    foreach ($it in $items) {
        while (@($running | Where-Object { -not $_.P.HasExited }).Count -ge $n) { Start-Sleep -Seconds 10 }
        $p = Start-Process -FilePath $Py -ArgumentList $it.Args -NoNewWindow -PassThru -WorkingDirectory $PWD.Path `
            -RedirectStandardOutput $it.Log -RedirectStandardError ($it.Log -replace "\.log$", ".err")
        $null = $p.Handle                       # keeps the exit code readable after the process ends
        [void]$running.Add(@{ Name = $it.Name; P = $p; Log = $it.Log })
        Write-Host ("  started {0,-28} {1:HH:mm}" -f $it.Name, (Get-Date))
    }
    while (@($running | Where-Object { -not $_.P.HasExited }).Count -gt 0) { Start-Sleep -Seconds 10 }
    foreach ($r in $running) {
        if ($r.P.ExitCode -ne 0) {
            Write-Host "  FAILED $($r.Name) (exit $($r.P.ExitCode)); last lines of its log:" -ForegroundColor Red
            Get-Content ($r.Log -replace "\.log$", ".err") -Tail 5 -ErrorAction SilentlyContinue
            Get-Content $r.Log -Tail 3 -ErrorAction SilentlyContinue
        }
    }
}
