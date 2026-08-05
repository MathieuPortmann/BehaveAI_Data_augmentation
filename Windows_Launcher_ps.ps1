<#
   BehaveAI - windows installer & launcher
  Windows_Launcher_ps.ps1 -- self-bootstrapping Ultralytics + venv launcher for Windows
  Usage:
    .\Windows_Launcher.bat                 # double-click or run from cmd
    powershell -ExecutionPolicy Bypass -NoProfile -File .\Windows_Launcher_ps.ps1 [script.py args...]
#>

param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $RemainingArgs
)

# Fail fast
$ErrorActionPreference = 'Stop'

# Logging/transcript
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$LogPath = Join-Path $ScriptDir "Windows_Launcher_ps.log"
if (Test-Path $LogPath) { Remove-Item $LogPath -ErrorAction SilentlyContinue }
Start-Transcript -Path $LogPath -Force

try {
    Write-Host "=== Windows_Launcher_ps.ps1 starting ==="

    # Config
    # Everything is self-contained next to this script so a freshly downloaded copy of
    # the repo works with no external files: the venv and requirements.txt both live in
    # the same folder as the launcher (and the BehaveAI .py code).
    $VENV_DIR = Join-Path $ScriptDir "BehaveAI.venv"
    $ReqFile = Join-Path $ScriptDir "requirements.txt"
    $PYTHON_CANDIDATES = @("py -3", "python", "python3")
    $MARKER = Join-Path $VENV_DIR ".ultralytics_ready"

    # -------------------------
    # Helper functions
    # -------------------------
    function Test-Command { param($cmd) try { & cmd /c "$cmd --version" > $null 2>&1; return $LASTEXITCODE -eq 0 } catch { return $false } }
    function Find-Python { foreach ($cmd in $PYTHON_CANDIDATES) { if (Test-Command $cmd) { return $cmd } }; return $null }

    function Ensure-Python {
        $found = Find-Python
        if ($found) { Write-Host "Found Python command: $found"; return $found }

        Write-Host ""
        $installChoice = Read-Host "Python 3 not found. Do you want to download & install Python 3.12 (64-bit)? (Y/N)"
        if ($installChoice.ToUpper() -ne 'Y') {
            Write-Host "User chose not to install Python. Aborting install."
            throw "Python missing"
        }

        Write-Host "Attempting to download and silently install Python 3.12 (64-bit)..." -ForegroundColor Yellow
        $pyVersion = "3.12.6"
        $url = "https://www.python.org/ftp/python/$pyVersion/python-$pyVersion-amd64.exe"
        $installer = Join-Path $env:TEMP "python-installer.exe"
        Write-Host "Downloading: $url"
        Invoke-WebRequest -Uri $url -OutFile $installer

        Write-Host "Running installer (silent, PrependPath=1). You may see a UAC prompt."
        $proc = Start-Process -FilePath $installer -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1" -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-Error "Python installer failed (exit code $($proc.ExitCode)). Please install Python manually."
            throw "Python installer failed"
        }

        Start-Sleep -Seconds 5
        $found = Find-Python
        if ($found) { Write-Host "Python installed and found as: $found"; return $found } else {
            Write-Warning "Python installed but not found in PATH. You may need to log out and back in."
            throw "Python not found after installation"
        }
    }

    function Venv-PythonExec { param([string[]]$Args) $p = Join-Path $VENV_DIR "Scripts\python.exe"; if (-not (Test-Path $p)) { throw "Venv python not found at $p" }; & $p @Args; return $LASTEXITCODE }

    function Detect-NvidiaGPU {
        try { $nvs = & nvidia-smi -L 2>$null; if ($LASTEXITCODE -eq 0 -and $nvs) { Write-Host "Detected NVIDIA GPU via nvidia-smi: $nvs"; return $true } } catch {}
        try { $adapters = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue; if ($adapters) { foreach ($a in $adapters) { if ($a.AdapterCompatibility -and $a.AdapterCompatibility -match "NVIDIA") { Write-Host "Detected NVIDIA GPU via WMI: $($a.Name)"; return $true } } } } catch {}
        return $false
    }

    # Map a CUDA version string (e.g. "12.8") reported by nvidia-smi to the closest
    # PyTorch wheel tag that actually ships a torch 2.10.0 build. Only cu126/cu128/cu130
    # exist for torch 2.10.0; older toolkits (cu118/cu121/cu124) have no 2.10.0 wheel.
    function Map-CudaVersionToWheel {
        param([string]$ver)
        if (-not $ver) { return $null }
        $v = $null
        if (-not [double]::TryParse($ver, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$v)) { return $null }
        if ($v -ge 13.0) { return @{ indexUrl = "https://download.pytorch.org/whl/cu130"; label="cu130" } }
        elseif ($v -ge 12.8) { return @{ indexUrl = "https://download.pytorch.org/whl/cu128"; label="cu128" } }
        elseif ($v -ge 12.6) { return @{ indexUrl = "https://download.pytorch.org/whl/cu126"; label="cu126" } }
        else { return $null }  # < 12.6: no compatible torch 2.10 CUDA wheel -> custom flow
    }

    # Open the PyTorch "get started" page and let the user paste the exact install
    # command. We normalise it (strip a leading pip/python -m pip) and run it in the venv.
    function Invoke-CustomTorchInstall {
        param([string]$venvPython)

        $url = "https://pytorch.org/get-started/locally/"
        Write-Host ""
        Write-Host "No compatible prebuilt CUDA wheel could be selected automatically." -ForegroundColor Yellow
        Write-Host "Opening the official PyTorch install page in your web browser:"
        Write-Host "  $url"
        try { Start-Process $url } catch { Write-Warning "Could not open browser automatically. Open this URL manually: $url" }

        Write-Host ""
        Write-Host "On that page, pick your OS / package=pip / your CUDA version, then COPY the"
        Write-Host "generated install command and PASTE it below (e.g.:"
        Write-Host "  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124 )."
        Write-Host ""
        $pasted = Read-Host "Paste the PyTorch install command (or press Enter to cancel)"
        if ([string]::IsNullOrWhiteSpace($pasted)) {
            Write-Warning "No command provided. Skipping custom PyTorch install."
            return $false
        }

        # Strip a leading pip / pip3 / python -m pip / python3 -m pip so we always run it
        # through the venv's interpreter.
        $argLine = $pasted.Trim()
        $argLine = [regex]::Replace($argLine, '^(python3?\s+-m\s+pip|pip3?)\s+', '', 'IgnoreCase')

        Write-Host ""
        Write-Host "The following command will run inside the venv:" -ForegroundColor Cyan
        Write-Host "  `"$venvPython`" -m pip $argLine"
        $confirm = Read-Host "Run it now? (Y/N)"
        if ($confirm.ToUpper() -ne 'Y') {
            Write-Host "Custom PyTorch install cancelled."
            return $false
        }

        # Split the argument line into tokens and invoke pip in the venv.
        $pipArgs = @("-m", "pip") + ([regex]::Matches($argLine, '[^\s]+') | ForEach-Object { $_.Value })
        & $venvPython @pipArgs
        if ($LASTEXITCODE -ne 0) { Write-Warning "Custom PyTorch install returned non-zero exit code ($LASTEXITCODE)."; return $false }
        return $true
    }

    # Returns a hashtable describing how torch should be installed:
    #   @{ mode = "index"; indexUrl = "..."; label = "cpu|cuNNN" }   -> pip install from index
    #   @{ mode = "custom" }                                          -> user pastes command
    function Choose-Torch-Wheel {
        param([bool]$nvidiaPresent)

        Write-Host ""
        Write-Host "PyTorch install options:"
        Write-Host "  1) CPU-only (recommended default)"
        Write-Host "  2) Auto-detect NVIDIA GPU and pick a compatible CUDA wheel (requires NVIDIA driver)"
        Write-Host "  3) Manually pick a CUDA wheel (cu126 / cu128 / cu130)"
        Write-Host "  4) Custom install via the PyTorch website (paste the command)"

        $choice = Read-Host "Choose option - 1=CPU, 2=Auto-detect, 3=Manual CUDA, 4=Custom. Enter 1/2/3/4 [default=2 auto]"
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "2" }

        switch ($choice) {
            "1" {
                return @{ mode="index"; indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" }
            }
            "2" {
                if (-not $nvidiaPresent) {
                    Write-Warning "No NVIDIA GPU detected. Falling back to CPU-only."
                    return @{ mode="index"; indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" }
                }

                Write-Host "Auto-detect: checking nvidia-smi for driver/CUDA info..."
                $ver = $null
                try {
                    $smi = & nvidia-smi 2>$null
                    if ($LASTEXITCODE -eq 0 -and $smi) {
                        $cudaLine = ($smi | Select-String -Pattern "CUDA Version" -SimpleMatch | Select-Object -First 1)
                        if ($cudaLine) {
                            $m = [regex]::Match($cudaLine.ToString(), "CUDA Version:\s*([0-9]+\.[0-9]+)")
                            if ($m.Success) { $ver = $m.Groups[1].Value; Write-Host "nvidia-smi reports CUDA $ver" }
                        }
                    }
                } catch {
                    $ver = $null
                }

                $wheel = Map-CudaVersionToWheel $ver
                if ($wheel) {
                    Write-Host "Auto-selected wheel: $($wheel.label)"
                    return @{ mode="index"; indexUrl = $wheel.indexUrl; label = $wheel.label }
                } else {
                    Write-Warning "No torch 2.10 CUDA wheel matches your driver (CUDA '$ver'). Switching to custom install."
                    return @{ mode="custom" }
                }
            }
            "3" {
                Write-Host ""
                Write-Host "Manual CUDA wheel choices (only tags with a torch 2.10.0 build):"
                Write-Host "  a) cu126 (CUDA 12.6)"
                Write-Host "  b) cu128 (CUDA 12.8)"
                Write-Host "  c) cu130 (CUDA 13.0)"
                Write-Host "  d) Custom (paste command from the PyTorch website)"
                $pick = Read-Host "Pick (a/b/c/d) or press Enter for cu128"
                switch ($pick) {
                    "a" { return @{ mode="index"; indexUrl = "https://download.pytorch.org/whl/cu126"; label="cu126" } }
                    "b" { return @{ mode="index"; indexUrl = "https://download.pytorch.org/whl/cu128"; label="cu128" } }
                    "c" { return @{ mode="index"; indexUrl = "https://download.pytorch.org/whl/cu130"; label="cu130" } }
                    "d" { return @{ mode="custom" } }
                    default { return @{ mode="index"; indexUrl = "https://download.pytorch.org/whl/cu128"; label="cu128" } }
                }
            }
            "4" {
                return @{ mode="custom" }
            }
            default {
                Write-Host "Unknown choice; defaulting to CPU-only."
                return @{ mode="index"; indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" }
            }
        }
    }

    function Is-Ready {
        if (Test-Path $MARKER) { return $true }
        $venvPython = Join-Path $VENV_DIR "Scripts\python.exe"
        if (Test-Path $venvPython) {
            try {
                & $venvPython -c "import ultralytics" > $null 2>&1; if ($LASTEXITCODE -ne 0) { return $false }
                & $venvPython -c "import cv2" > $null 2>&1; return ($LASTEXITCODE -eq 0)
            } catch { return $false }
        }
        return $false
    }

    # -------------------------
    # Bootstrap (create venv & install)
    # -------------------------
    function Bootstrap {
        Write-Host "== Ultralytics bootstrap for Windows: installing python packages into venv =="

        # find or install python
        $pyCmd = Ensure-Python

        Write-Host "Using Python launcher: $pyCmd"
        # Print the exact python version found
        try {
            $pyVer = & cmd /c "$pyCmd --version" 2>&1
            Write-Host "Python version: $pyVer"
        } catch {
            Write-Warning "Couldn't determine Python version with '$pyCmd --version'."
        }

        Write-Host "Creating virtualenv at $VENV_DIR (if missing)..."
        if (-not (Test-Path $VENV_DIR)) {
            & cmd /c "$pyCmd -m venv `"$VENV_DIR`""
            if ($LASTEXITCODE -ne 0) { throw "Failed to create virtualenv" }
        } else { Write-Host "Virtualenv already exists - reusing." }

        $venvPython = Join-Path $VENV_DIR "Scripts\python.exe"
        if (-not (Test-Path $venvPython)) { throw "Venv python not present after creation" }

        Write-Host "Virtualenv python path: $venvPython"
        try {
            $venvPyVer = & $venvPython --version 2>&1
            Write-Host "Virtualenv Python version: $venvPyVer"
        } catch {
            Write-Warning "Could not run venv python --version."
        }

        Write-Host "Upgrading pip, setuptools, wheel inside venv..."
        & $venvPython -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { Write-Warning "pip upgrade reported non-zero exit code" }

        # Ask CPU/GPU choice here
        Write-Host ""
        $installLibsChoice = Read-Host "Install ultralytics, torch and required Python packages now? (Y/N)"
        if ($installLibsChoice.ToUpper() -ne 'Y') { Write-Host "Skipping package installation per user request."; return }

        Write-Host "Checking for NVIDIA GPU..."
        $hasNvidia = Detect-NvidiaGPU
        $torchChoice = Choose-Torch-Wheel -nvidiaPresent:$hasNvidia

        # 1) Install the project dependencies from requirements.txt FIRST. ultralytics
        #    pulls a default (CPU) torch here; that is fine because we (re)install the
        #    machine-specific torch build LAST so it always wins. (Installing torch before
        #    requirements lets pip's transitive re-resolution downgrade it back to CPU.)
        if (-not (Test-Path $ReqFile)) { throw "requirements.txt not found at $ReqFile" }
        Write-Host "Installing project dependencies from requirements.txt..."
        # requirements.txt does not list torch/torchvision, but filter defensively in case.
        $reqLines = Get-Content $ReqFile | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio)\b' }
        $tmpReq = Join-Path $env:TEMP "behaveai_requirements_filtered.txt"
        $reqLines | Set-Content -Encoding ascii $tmpReq
        & $venvPython -m pip install -r $tmpReq
        if ($LASTEXITCODE -ne 0) { throw "pip install of requirements.txt failed" }
        Remove-Item $tmpReq -ErrorAction SilentlyContinue

        # 2) Install the machine-specific PyTorch LAST so it is the final, authoritative build.
        try {
            if ($torchChoice.mode -eq "custom") {
                # User pastes the exact install command from the PyTorch website.
                [void](Invoke-CustomTorchInstall -venvPython $venvPython)
            } else {
                Write-Host "Installing PyTorch ($($torchChoice.label)) - version chosen by your machine/CUDA, not requirements.txt..."
                # --upgrade so it replaces any CPU torch pulled above; --no-deps so it cannot
                # disturb the pinned requirements packages (numpy, etc.) - torch's runtime
                # deps are already provided by requirements.txt. torchaudio omitted (unused).
                & $venvPython -m pip install --upgrade --no-deps --index-url $torchChoice.indexUrl torch torchvision
            }
            if ($LASTEXITCODE -ne 0) { Write-Warning "PyTorch install returned non-zero exit code; you may need to retry manually." }
        } catch {
            Write-Warning "PyTorch installation raised an error: $_"
        }

        Write-Host ""
        Write-Host "Verifying important imports inside the venv (this may take a moment)..."

        # ultralytics
        try {
            & $venvPython -c "import ultralytics; print('ultralytics OK')" | Out-Host
            Write-Host "ultralytics import: SUCCESS"
        } catch {
            Write-Warning "ultralytics import: FAILED - check the log for details."
        }

        # torch
        try {
            & $venvPython -c "import torch; print('torch OK', torch.__version__)" | Out-Host
            Write-Host "torch import: SUCCESS"
        } catch {
            Write-Warning "torch import: FAILED - if you requested CUDA, confirm your NVIDIA driver and chosen CUDA wheel."
        }

        # cv2
        try {
            & $venvPython -c "import cv2; print('cv2 OK', cv2.__version__)" | Out-Host
            Write-Host "cv2 import: SUCCESS"
        } catch {
            Write-Warning "cv2 import: FAILED - consider installing opencv-contrib-python or check the log."
        }

        # marker
        if (-not (Test-Path $VENV_DIR)) { New-Item -ItemType Directory -Path $VENV_DIR | Out-Null }
        New-Item -ItemType File -Force -Path $MARKER | Out-Null
        Write-Host "Bootstrap complete."
    }

    # -------------------------
    # Main flow: ask user if need to install, unless env is already ready
    # -------------------------
    $envReady = Is-Ready
    if (-not $envReady) {
        Write-Host ""
        $installPrompt = Read-Host "Environment not ready. Install Python + libraries and create venv now? (Y/N)"
        if ($installPrompt.ToUpper() -eq 'Y') {
            Bootstrap
            $envReady = Is-Ready
            if (-not $envReady) { Write-Warning "Environment still not ready after bootstrap. Check the log: $LogPath" }
        } else {
            Write-Host "User chose not to install. If environment is not ready, the script will now exit."
            Stop-Transcript
            exit 2
        }
    } else {
        Write-Host "Environment already ready. Using existing venv at $VENV_DIR"
    }

    # -------------------------
    # Run requested script (or BehaveAI.py by default)
    # -------------------------
    $venvPythonExe = Join-Path $VENV_DIR "Scripts\python.exe"
    if ($RemainingArgs -and $RemainingArgs.Count -gt 0) {
        Write-Host "Running: $venvPythonExe $($RemainingArgs -join ' ')"
        & $venvPythonExe @RemainingArgs
        $exitCode = $LASTEXITCODE
        Write-Host "Script exited with code $exitCode"
        Stop-Transcript
        exit $exitCode
    } else {
        # Default: run BehaveAI.py shipped next to this launcher (works no matter which
        # folder the user double-clicks from).
        $default = Join-Path $ScriptDir "BehaveAI.py"
        if (Test-Path $default) {
            Write-Host "Launching BehaveAI ($default)"
            & $venvPythonExe $default
            $exitCode = $LASTEXITCODE
            Write-Host "BehaveAI.py exited with code $exitCode"
            Stop-Transcript
            exit $exitCode
        } else {
            Write-Host "BehaveAI.py not found next to the launcher ($ScriptDir)." -ForegroundColor Yellow
            Write-Host "Usage: Windows_Launcher.bat path\to\script.py [args...]"
            Stop-Transcript
            exit 2
        }
    }
}
catch {
    Write-Error "Fatal error: $_"
    Write-Host "See the log at: $LogPath"
    Stop-Transcript
    exit 1
}
