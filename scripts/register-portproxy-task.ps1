# register-portproxy-task.ps1
# Run once (as Administrator) to register the startup task in Task Scheduler.
# After registration, the proxy updates automatically on every boot.

#Requires -RunAsAdministrator

$taskName  = "Vision RAG - WSL2 Port Proxy"
$scriptPath = "\\wsl.localhost\Ubuntu\home\appel\projects\Vision_RAG\scripts\update-wsl-portproxy.ps1"

# Action: run the update script hidden
$action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`""

# Trigger: at system startup (runs before any user logs in)
$trigger = New-ScheduledTaskTrigger -AtStartup

# Small delay so the network stack is up before we query WSL2
$trigger.Delay = "PT10S"   # 10-second delay

# Run as SYSTEM so it's always elevated, no UAC prompt
$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable          # catch up if system was off at scheduled time

# Register (overwrites if it already exists)
Register-ScheduledTask `
    -TaskName  $taskName `
    -Action    $action `
    -Trigger   $trigger `
    -Principal $principal `
    -Settings  $settings `
    -Force | Out-Null

Write-Host "Task '$taskName' registered successfully." -ForegroundColor Green
Write-Host "Running it now for the first time..."

Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 5
$state = (Get-ScheduledTask -TaskName $taskName).State
Write-Host "Task state: $state"

Write-Host ""
Write-Host "Log file: $env:TEMP\vision-rag-portproxy.log"
Write-Host "Vision RAG will be available at http://10.0.0.6:3000 after each reboot."
