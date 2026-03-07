# update-wsl-portproxy.ps1
# Reads the current WSL2 IP and updates netsh portproxy rules for Vision RAG.
# Runs at startup via Task Scheduler (as SYSTEM / elevated).

$ports     = @(3000, 8081, 8082)
$ruleName  = "Vision RAG WSL2"
$logFile   = "$env:TEMP\vision-rag-portproxy.log"

function Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$ts  $msg" | Tee-Object -FilePath $logFile -Append | Write-Host
}

Log "=== Vision RAG port-proxy update ==="

# ── 1. Wait for WSL2 to be ready ─────────────────────────────────────────────
$wslIp = $null
for ($i = 0; $i -lt 20; $i++) {
    try {
        $raw = & wsl.exe hostname -I 2>$null
        $wslIp = ($raw -split '\s+' | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1)
    } catch {}
    if ($wslIp) { break }
    Log "Waiting for WSL2 to start (attempt $($i+1)/20)..."
    Start-Sleep -Seconds 3
}

if (-not $wslIp) {
    Log "ERROR: Could not get WSL2 IP after 60 s — aborting."
    exit 1
}
Log "WSL2 IP: $wslIp"

# ── 2. Remove stale portproxy entries ────────────────────────────────────────
foreach ($port in $ports) {
    netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>$null | Out-Null
}

# ── 3. Add fresh portproxy entries ───────────────────────────────────────────
foreach ($port in $ports) {
    netsh interface portproxy add v4tov4 `
        listenport=$port listenaddress=0.0.0.0 `
        connectport=$port connectaddress=$wslIp | Out-Null
    Log "Forwarded 0.0.0.0:$port -> $wslIp:$port"
}

# ── 4. Ensure firewall rule exists (idempotent) ───────────────────────────────
$fwExists = netsh advfirewall firewall show rule name=$ruleName 2>$null
if ($LASTEXITCODE -ne 0) {
    netsh advfirewall firewall add rule `
        name=$ruleName dir=in action=allow protocol=tcp `
        localport=($ports -join ',') | Out-Null
    Log "Firewall rule '$ruleName' created."
} else {
    Log "Firewall rule '$ruleName' already exists."
}

Log "Done. Vision RAG available at http://10.0.0.6:3000"
