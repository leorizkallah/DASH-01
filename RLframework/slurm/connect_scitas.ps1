# Run in a visible terminal: the password is entered directly into OpenSSH.
# Creates a dedicated key without altering existing keys or SSH configuration.
$ErrorActionPreference = 'Stop'
$taskKey = Join-Path $env:USERPROFILE '.ssh/dash_scitas_training_ed25519'
if (-not (Test-Path -LiteralPath $taskKey)) {
    & ssh-keygen -t ed25519 -f $taskKey -N '""' -C 'dash-scitas-training'
    if ($LASTEXITCODE -ne 0) { throw 'SSH key generation failed.' }
}
$taskPublic = (Get-Content -LiteralPath ($taskKey + '.pub') -Raw).Trim()
if ($taskPublic -notmatch '^ssh-ed25519 [A-Za-z0-9+/=]+ dash-scitas-training$') {
    throw 'Unexpected public-key format; nothing was installed.'
}
$taskRemote = "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -qxF '$taskPublic' ~/.ssh/authorized_keys || printf '%s\n' '$taskPublic' >> ~/.ssh/authorized_keys"
Write-Host 'Enter your SCITAS password below. It will not be displayed or saved by this script.'
& ssh -o ConnectTimeout=15 rizkalla@izar.hpc.epfl.ch $taskRemote
if ($LASTEXITCODE -ne 0) { throw 'SCITAS login/key installation failed.' }
& ssh -i $taskKey -o IdentitiesOnly=yes -o BatchMode=yes rizkalla@izar.hpc.epfl.ch 'hostname'
if ($LASTEXITCODE -ne 0) { throw 'Key authentication verification failed.' }
Write-Host 'SCITAS key access is ready. You can close this window.'
