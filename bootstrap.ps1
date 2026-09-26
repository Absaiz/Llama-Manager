# bootstrap.ps1 - prepara .\python (Python embebido + Flask) y el motor Vulkan.
# Lo lanza LlamaManager.bat la primera vez. Se puede repetir sin problema.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'      # Invoke-WebRequest es 10x mas lento con la barra
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location $PSScriptRoot

$v   = Get-Content (Join-Path $PSScriptRoot 'versions.py') -Raw
$ver = [regex]::Match($v, 'PYTHON_VERSION\s*=\s*"([^"]+)"').Groups[1].Value
$tag = [regex]::Match($v, 'PYTHON_STANDALONE_TAG\s*=\s*"([^"]+)"').Groups[1].Value
if (-not $ver -or -not $tag) { throw 'No encuentro PYTHON_VERSION / PYTHON_STANDALONE_TAG en versions.py' }

$py = Join-Path $PSScriptRoot 'python\python.exe'
if (-not (Test-Path $py)) {
    $url = "https://github.com/astral-sh/python-build-standalone/releases/download/$tag/cpython-$ver+$tag-x86_64-pc-windows-msvc-install_only.tar.gz"
    $tgz = Join-Path $env:TEMP "llama-manager-python-$ver.tar.gz"
    Write-Host "Descargando Python $ver ..."
    Invoke-WebRequest -Uri $url -OutFile $tgz -UseBasicParsing
    Write-Host 'Descomprimiendo ...'
    tar -xzf $tgz -C $PSScriptRoot           # crea .\python
    if ($LASTEXITCODE -ne 0) { throw "tar fallo ($LASTEXITCODE)" }
    Remove-Item $tgz -Force
    # fuera simbolos de depuracion y Tk/IDLE (~80 MB que no se usan)
    Get-ChildItem (Join-Path $PSScriptRoot 'python') -Recurse -Filter *.pdb | Remove-Item -Force
    foreach ($d in 'include','tcl','Lib\idlelib','Lib\tkinter','Lib\turtledemo','Lib\test') {
        $p = Join-Path $PSScriptRoot "python\$d"
        if (Test-Path $p) { Remove-Item $p -Recurse -Force }
    }
}

Write-Host 'Instalando Flask ...'
& $py -m pip install --disable-pip-version-check --no-warn-script-location --quiet flask
if ($LASTEXITCODE -ne 0) { throw 'pip no pudo instalar flask (sin internet?)' }
# opcionales: nucleos fisicos en el log (psutil) y busqueda web del agente (ddgs)
& $py -m pip install --disable-pip-version-check --no-warn-script-location --quiet psutil ddgs
if ($LASTEXITCODE -ne 0) { Write-Host 'Aviso: psutil/ddgs no instalados (opcionales)' }

if (-not (Test-Path (Join-Path $PSScriptRoot 'llama-vulkan\llama-server.exe'))) {
    Write-Host 'Descargando llama.cpp (Vulkan) ...'
    & $py (Join-Path $PSScriptRoot 'update_llama.py') vulkan
}
Write-Host 'Listo.'
exit 0
