# Refresh the experiment's measured report and download its small result files.
param(
    [string]$Snapshot = 'stop01_20261002_v2',
    [string]$Run = 'dash_stop01b_s0',
    [string]$Job = '3200768',
    [string]$LogPrefix = 'dash-stop01'
)
$ErrorActionPreference = 'Stop'
if ($Snapshot -notmatch '^[a-zA-Z0-9_-]+$' -or $Run -notmatch '^[a-zA-Z0-9_-]+$' -or $Job -notmatch '^\d+$' -or $LogPrefix -notmatch '^[a-zA-Z0-9_-]+$') {
    throw 'Invalid experiment identifiers.'
}
$taskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskKey = Join-Path $env:USERPROFILE '.ssh/dash_scitas_training_ed25519'
$taskHostName = 'rizkalla@izar.hpc.epfl.ch'
$taskRemoteRoot = "/home/rizkalla/running_robot_experiments/$Snapshot"
$taskArgs = @('-i', $taskKey, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes')
& ssh @taskArgs $taskHostName "cd $taskRemoteRoot && /home/rizkalla/venvs/dash-v2/bin/python RLframework/tools/stop_report.py --run RLframework/runs/$Run; squeue -j $Job; sacct -j $Job --format=JobID,State,Elapsed,ExitCode; tail -25 $taskRemoteRoot/RLframework/logs/$LogPrefix-$Job.out"
if ($LASTEXITCODE -ne 0) { throw 'Remote progress query failed.' }
$taskDest = Join-Path $taskRoot "docs/experiments/$Snapshot"
New-Item -ItemType Directory -Force -Path $taskDest | Out-Null
foreach ($taskFile in @('STOP_PROGRESS.md', 'warmstart.json', 'critic_prefit.json', 'imitation.json', 'stop_baseline.json', 'stop_evaluations.jsonl', 'best_stop_eval.json', 'progress.csv', 'eval.csv', 'resolved_config.json')) {
    # Missing files mean that this stage has not completed yet; preserve earlier downloads.
    & ssh @taskArgs $taskHostName "test -f $taskRemoteRoot/RLframework/runs/$Run/$taskFile"
    if ($LASTEXITCODE -eq 0) {
        & scp @taskArgs "${taskHostName}:$taskRemoteRoot/RLframework/runs/$Run/$taskFile" $taskDest
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $taskFile" }
    }
}
$taskCsv = Join-Path $taskDest 'progress.csv'
if (Test-Path -LiteralPath $taskCsv) {
    $taskLast = Import-Csv -LiteralPath $taskCsv | Select-Object -Last 1
    if ($taskLast) {
        Add-Content -LiteralPath (Join-Path $taskDest 'STOP_PROGRESS.md') -Value "`nLatest recorded PPO environment steps: $($taskLast.'time/env_steps')."
    }
}
Write-Host "Progress files saved to $taskDest"
