@powershell -NoProfile -ExecutionPolicy Unrestricted "$s=[scriptblock]::create((gc \"%~f0\"|?{$_.readcount -gt 1})-join\"`n\");&$s "%~dp0 %*&goto:eof
Write-Host $args

$base_dir = Split-Path $args -Parent
$dist_dir = Join-Path $base_dir "\dist"
Set-Location -path $dist_dir

$param_json_file = Join-Path $args "params.json"
$target_path = Join-Path $base_dir "pref\poc3.py"
Write-Host $base_dir
Write-Host $target_path
[string[]]$args_list = @($target_path, "--onefile", "--noconsole", "--distpath=./win", "--log-level DEBUG")

#json ファイルからpyinstallerのファイルパスを取得
$json = Get-Content $param_json_file -Encoding UTF8 -Raw | ConvertFrom-Json
Write-Host "Start!!"
Write-Host $json.pyinstaller

Write-Output $args_list
Start-Process -FilePath $json.pyinstaller -ArgumentList $args_list -Wait

Write-Host "Done!!"
Start-Sleep -s 10