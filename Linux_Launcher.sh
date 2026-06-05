#!/usr/bin/env bash
# run-yolo.sh -- self-bootstrapping wrapper for Ultralytics + apt OpenCV on Raspberry Pi OS
# Usage:
#   ./run-yolo.sh             # if BehaveAI.py exists in CWD, runs it
#   ./run-yolo.sh script.py [args...]  # runs a specific script with args
set -euo pipefail

# --- Config ---
# Everything is self-contained next to this script so a freshly downloaded copy of the
# repo works with no external files: the venv and requirements.txt both live in the same
# folder as this launcher (and the BehaveAI .py code).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/BehaveAI.venv"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
PYTHON_BIN="/usr/bin/python3"
# List of apt packages we may need
APT_PKGS=(python3-venv python3-pip build-essential git wget curl ffmpeg \
          libglib2.0-0 libsm6 libxrender1 libxext6 libjpeg-dev zlib1g-dev \
          python3-opencv)
# A tiny marker file to indicate successful install (optional but nice)
MARKER="${VENV_DIR}/.ultralytics_ready"

# Helper: run python inside venv (without permanently activating)
venv_python() {
  # use the venv python if present
  if [ -x "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

# Check whether venv + ultralytics are already installed & usable
is_ready() {
  if [ -f "${MARKER}" ]; then
    return 0
  fi
  # if venv exists, try to import ultralytics from it
  if [ -x "${VENV_DIR}/bin/python" ]; then
    if "${VENV_DIR}/bin/python" -c "import ultralytics, sys; sys.exit(0)" >/dev/null 2>&1; then
      # also check cv2 is importable (should be via system-site-packages)
      if "${VENV_DIR}/bin/python" -c "import cv2, sys; sys.exit(0)" >/dev/null 2>&1; then
        return 0
      fi
    fi
  fi
  return 1
}

# Map a CUDA version (e.g. "12.8") reported by nvidia-smi to the closest PyTorch wheel
# tag that ships a torch 2.10.0 build. Only cu126/cu128/cu130 exist for torch 2.10.0.
# Prints the index URL on success, prints nothing (returns 1) when no wheel matches.
map_cuda_to_wheel() {
  local ver="$1"
  [ -z "${ver}" ] && return 1
  # integer comparison on major*100+minor avoids needing bc
  local key
  key=$(awk -v v="${ver}" 'BEGIN{split(v,a,"."); printf "%d", a[1]*100 + a[2]}')
  if   [ "${key}" -ge 1300 ]; then echo "https://download.pytorch.org/whl/cu130"
  elif [ "${key}" -ge 1208 ]; then echo "https://download.pytorch.org/whl/cu128"
  elif [ "${key}" -ge 1206 ]; then echo "https://download.pytorch.org/whl/cu126"
  else return 1
  fi
}

# Open the PyTorch site and let the user paste the install command; run it in the venv.
custom_torch_install() {
  local py="$1"
  local url="https://pytorch.org/get-started/locally/"
  echo
  echo "No compatible prebuilt CUDA wheel could be selected automatically."
  echo "Opening the official PyTorch install page: ${url}"
  xdg-open "${url}" >/dev/null 2>&1 || echo "  (Open this URL manually in a browser: ${url})"
  echo
  echo "On that page pick your OS / package=pip / your CUDA version, COPY the generated"
  echo "command and PASTE it below (e.g. pip3 install torch torchvision --index-url ...)."
  echo
  read -r -p "Paste the PyTorch install command (or press Enter to cancel): " pasted
  if [ -z "${pasted}" ]; then echo "No command provided. Skipping custom PyTorch install."; return 1; fi
  # Strip a leading pip / pip3 / python -m pip so we always run via the venv interpreter.
  local args
  args="$(echo "${pasted}" | sed -E 's/^(python3?[[:space:]]+-m[[:space:]]+pip|pip3?)[[:space:]]+//I')"
  echo
  echo "The following command will run inside the venv:"
  echo "  ${py} -m pip ${args}"
  read -r -p "Run it now? (Y/N): " confirm
  case "${confirm}" in
    [Yy]*) ;;
    *) echo "Custom PyTorch install cancelled."; return 1 ;;
  esac
  # shellcheck disable=SC2086
  "${py}" -m pip ${args}
}

# Present the torch install menu and install torch/torchvision accordingly.
install_torch() {
  local py="$1"
  echo
  echo "PyTorch install options:"
  echo "  1) CPU-only (default)"
  echo "  2) Auto-detect NVIDIA GPU and pick a compatible CUDA wheel"
  echo "  3) Manually pick a CUDA wheel (cu126 / cu128 / cu130)"
  echo "  4) Custom install via the PyTorch website (paste the command)"
  read -r -p "Choose 1/2/3/4 [default=2 auto]: " choice
  [ -z "${choice}" ] && choice="2"

  local idx=""
  case "${choice}" in
    1) idx="https://download.pytorch.org/whl/cpu" ;;
    2)
      local ver=""
      if command -v nvidia-smi >/dev/null 2>&1; then
        ver="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | grep -oE '[0-9]+\.[0-9]+')"
      fi
      if [ -z "${ver}" ]; then
        # No NVIDIA driver/GPU (e.g. Raspberry Pi or CPU-only host) -> CPU build.
        echo "No NVIDIA GPU/driver detected -> installing the CPU build."
        idx="https://download.pytorch.org/whl/cpu"
      else
        echo "nvidia-smi reports CUDA ${ver}"
        idx="$(map_cuda_to_wheel "${ver}")" || idx=""
        if [ -z "${idx}" ]; then
          echo "No prebuilt CUDA wheel matches your driver (CUDA '${ver}'). Switching to custom install."
          custom_torch_install "${py}"; return $?
        fi
        echo "Auto-selected: ${idx##*/}"
      fi
      ;;
    3)
      echo "  a) cu126   b) cu128   c) cu130   d) Custom"
      read -r -p "Pick (a/b/c/d) [default=b]: " pick
      case "${pick}" in
        a) idx="https://download.pytorch.org/whl/cu126" ;;
        c) idx="https://download.pytorch.org/whl/cu130" ;;
        d) custom_torch_install "${py}"; return $? ;;
        *) idx="https://download.pytorch.org/whl/cu128" ;;
      esac
      ;;
    4) custom_torch_install "${py}"; return $? ;;
    *) idx="https://download.pytorch.org/whl/cpu" ;;
  esac

  echo "Installing PyTorch (${idx##*/}) - version chosen by your machine/CUDA, not requirements.txt..."
  # Unpinned on purpose: pip resolves the latest torch/torchvision for the selected index
  # (cpu/cuNNN) so the build matches this machine. torchaudio omitted (not imported).
  "${py}" -m pip install --index-url "${idx}" torch torchvision
}

bootstrap() {
  echo "== Ultralytics bootstrap: installing system & python dependencies =="
  echo "You may be asked for your sudo password to install apt packages."
  # Update and install apt packages
  sudo apt update
  sudo apt install -y "${APT_PKGS[@]}"

  # Create venv (with system site packages so apt OpenCV is visible)
  if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtualenv at ${VENV_DIR} (with --system-site-packages)..."
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    echo "Virtualenv already exists at ${VENV_DIR} - reusing."
  fi

  # Ensure pip in venv is up-to-date and install pip packages
  echo "Upgrading pip and installing Python packages inside venv..."
  # shellcheck disable=SC1090
  # Use a subshell so activation doesn't pollute caller environment
  (
    set -e
    source "${VENV_DIR}/bin/activate"
    python -m pip install --upgrade pip setuptools wheel

    # 1) Install torch/torchvision (CPU / CUDA / custom) via the interactive menu.
    install_torch "${VENV_DIR}/bin/python"

    # 2) Install the rest of the project dependencies from requirements.txt (the source of
    #    truth). Exclude torch/torchvision/torchaudio (handled above) and opencv-python
    #    (provided by apt's python3-opencv via --system-site-packages).
    if [ ! -f "${REQ_FILE}" ]; then
      echo "ERROR: requirements.txt not found at ${REQ_FILE}" >&2
      exit 1
    fi
    echo "Installing project dependencies from requirements.txt (excluding torch/vision/opencv)..."
    # NOTE: on ARM (Raspberry Pi) some packages (scipy, scikit-learn, matplotlib) may build
    # from source and take a while.
    grep -viE '^[[:space:]]*(torch|torchvision|torchaudio|opencv-python)\b' "${REQ_FILE}" > /tmp/behaveai_req_filtered.txt
    python -m pip install -r /tmp/behaveai_req_filtered.txt
    rm -f /tmp/behaveai_req_filtered.txt
  )

  # final sanity checks
  if ! "${VENV_DIR}/bin/python" -c "import ultralytics" >/dev/null 2>&1; then
    echo "ERROR: ultralytics import failed after pip install." >&2
    exit 1
  fi
  if ! "${VENV_DIR}/bin/python" -c "import cv2" >/dev/null 2>&1; then
    echo "WARNING: OpenCV (cv2) not importable inside venv. You may need to install python3-opencv via apt." >&2
    # We continue, since apt install was attempted above; user can re-run
  fi

  # create marker
  mkdir -p "${VENV_DIR}"
  touch "${MARKER}"
  echo "Bootstrap complete."
  echo
}

# If not ready, bootstrap (this will be performed only the first time or if something missing)
if ! is_ready; then
  bootstrap
fi

# Activate venv for the remainder of this script (so python uses venv)
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Now run the requested python script.
if [ "$#" -ge 1 ]; then
  # Run args as python command
  exec python "$@"
else
  # No args: run BehaveAI.py shipped next to this launcher (works regardless of CWD).
  if [ -f "${SCRIPT_DIR}/BehaveAI.py" ]; then
    echo "Launching BehaveAI (${SCRIPT_DIR}/BehaveAI.py)"
    exec python "${SCRIPT_DIR}/BehaveAI.py"
  else
    cat <<EOF
BehaveAI.py not found next to the launcher (${SCRIPT_DIR}).

Usage:
  ${0} path/to/script.py [args...]

EOF
    exit 2
  fi
fi
