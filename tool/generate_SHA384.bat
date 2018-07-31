@powershell -NoProfile -ExecutionPolicy Unrestricted "$s=[scriptblock]::create((gc \"%~f0\"|?{$_.readcount -gt 1})-join\"`n\");&$s "%~df0 %*&goto:eof
Write-Host $args
Get-FileHash $args -Algorithm SHA384
Start-Sleep -s 10