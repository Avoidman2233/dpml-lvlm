#!/bin/bash
# Cloud server environment setup – no venv, installs dassl from Dassl.pytorch/
# Usage: bash scripts/setup_env.sh [--cuda 11.8|12.1] [--dry-run]
set -e

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
CUDA_VER="11.8"
DRY_RUN=false
PYTHON="python3"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)    CUDA_VER="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *)         echo "⚠ Unknown arg: $1"; shift;;
    esac
done

command_exists() { command -v "$1" &>/dev/null; }
pip_pkg_exists() { $PYTHON -c "import $1" &>/dev/null; }

run() {
    echo "→ $*"
    $DRY_RUN || eval "$@"
}

BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m";  RED="\033[31m"; NC="\033[0m"
step()  { echo -e "${BOLD}[STEP]${NC} $1"; }
ok()    { echo -e "  ${GREEN}✓${NC} $1"; }
skip()  { echo -e "  ${YELLOW}○${NC} $1 (already satisfied)"; }
fail()  { echo -e "  ${RED}✗${NC} $1"; exit 1; }

echo -e "${BOLD}ATPrompt – Cloud Server Environment Setup${NC}"
echo "  Project : $PROJ"
echo "  CUDA    : $CUDA_VER"
echo "  Dry-run : $DRY_RUN"
echo ""

# == 1. Check Python =========================================================
step "1/6  Checking Python…"
if ! command_exists "$PYTHON"; then
    PYTHON="python"
    command_exists "$PYTHON" || fail "Python not found. Install Python >= 3.8 first."
fi
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
ok "Python $PY_VER"

# == 2. CUDA toolkit check ===================================================
step "2/6  Checking CUDA…"
if command_exists nvcc; then
    NVCC_VER=$(nvcc --version | grep "release" | awk -F'[, ]' '{print $6}' | sed 's/V//')
    ok "CUDA $NVCC_VER (nvcc)"
elif command_exists nvidia-smi; then
    DRV_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    ok "CUDA $DRV_VER (driver, no nvcc)"
else
    echo "  ${YELLOW}⚠${NC} No CUDA detected — will install CPU-only PyTorch"
    CUDA_VER="cpu"
fi

# == 3. Install PyTorch (CUDA-aware) ==========================================
step "3/6  Installing PyTorch…"
if pip_pkg_exists torch; then
    skip "torch"
else
    case "$CUDA_VER" in
        12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        cpu)   TORCH_INDEX="" ;;
        *)     TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    esac
    if [ -z "$TORCH_INDEX" ]; then
        run pip install torch torchvision
    else
        run pip install torch torchvision --index-url "$TORCH_INDEX"
    fi
    ok "PyTorch installed (CUDA $CUDA_VER)"
fi

# == 4. Install pip requirements =============================================
step "4/6  Installing pip dependencies…"
run pip install -r "$PROJ/requirements.txt"
ok "requirements.txt done"

# == 5. Install dassl from local Dassl.pytorch/ ================================
step "5/6  Installing dassl (from Dassl.pytorch/)…"
DASSL_DIR="$PROJ/Dassl.pytorch"
if [ ! -d "$DASSL_DIR" ]; then
    fail "Dassl.pytorch/ not found at $DASSL_DIR"
fi
run pip install -e "$DASSL_DIR"
ok "dassl installed (editable)"

# == 6. Extra packages ======================================================
step "6/6  Installing remaining packages…"
run pip install open_clip_torch python-dateutil
ok "extra packages done"

# == Verification ===========================================================
echo ""
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${BOLD}  Verification${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"

verify() {
    local pkg="$1"
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $pkg"
    else
        echo -e "  ${RED}✗${NC} $pkg – WARNING"
    fi
}

verify torch
verify torchvision
verify dassl
verify clip
verify numpy
verify tqdm

echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "  ${GREEN}Environment setup complete.${NC}"
echo -e "  Run   ${BOLD}bash scripts/dlmpt/train.sh --dry-run${NC}   to test."
echo -e "${BOLD}──────────────────────────────────────────${NC}"
