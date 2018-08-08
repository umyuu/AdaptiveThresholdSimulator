@powershell -NoProfile -ExecutionPolicy Unrestricted "$s=[scriptblock]::create((gc \"%~f0\"|?{$_.readcount -gt 1})-join\"`n\");&$s "%~dp0 %*&goto:eof
Write-Host $args

$base_dir = Split-Path $args -Parent
$src_dir = Join-Path $base_dir "\src"
$dist_dir = Join-Path $base_dir "\dist"
Set-Location -path $dist_dir

$param_json_file = Join-Path $args "params.json"
#$target_path = Join-Path $dist_dir "src\simulator.py"
$target_path = Join-Path $src_dir "simulator.py"

Write-Host $base_dir
Write-Host $target_path

#json ファイルからpyinstallerのファイルパスを取得
$json = Get-Content $param_json_file -Encoding UTF8 -Raw | ConvertFrom-Json
Write-Host "Start!!"
# @see https://pyinstaller.readthedocs.io/en/v3.3.1/usage.html#general-options
[string[]]$args_list = @($target_path, "-y", "--clean", "-d", "--distpath=./win", "--log-level DEBUG", "--add-data ../src/MainWindow.xml;.")

Write-Output $args_list
#Set-Location -path $src_dir
Start-Process -FilePath $json.pyinstaller -ArgumentList $args_list -Wait

Write-Host "Done!!"
Start-Sleep -s 7