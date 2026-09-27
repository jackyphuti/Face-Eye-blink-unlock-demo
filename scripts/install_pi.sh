#!/usr/bin/env bash
# ==============================================================================
# Raspberry Pi Automated Installation Script for Face-Eye-Blink-Unlock
# Compatible with Raspberry Pi OS (Debian Bullseye / Bookworm on Pi 3, 4, 5, Zero 2W)
# ==============================================================================

set -e

echo "=========================================================="
echo " Starting Raspberry Pi Face-Eye-Blink Biometric Setup"
echo "=========================================================="

# Check if running as non-root with sudo privileges
if [ "$EUID" -eq 0 ]; then
  echo "Please run this script as a normal user with sudo privileges (not root)."
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] Updating APT repositories..."
sudo apt update

echo "[2/6] Installing native build and system dependencies..."
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libopenblas-dev \
  liblapack-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libxvidcore-dev \
  libx264-dev \
  libgtk-3-dev \
  python3-dev \
  python3-pip \
  python3-venv \
  python3-rpi.gpio \
  python3-gpiozero \
  python3-serial \
  v4l-utils

# Check total RAM. If < 2GB (Pi 3 or Zero 2W), expand swap file to prevent OOM
TOTAL_MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
echo "Detected RAM: ${TOTAL_MEM_MB} MB"

if [ "$TOTAL_MEM_MB" -lt 1800 ]; then
  echo "Detected low RAM device (<2GB). Ensuring swap size is at least 1500MB for dlib installation..."
  if [ -f /etc/dphys-swapfile ]; then
    CURRENT_SWAP=$(grep "^CONF_SWAPSIZE=" /etc/dphys-swapfile | cut -d= -f2)
    if [ "$CURRENT_SWAP" -lt 1500 ]; then
      sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1500/' /etc/dphys-swapfile
      sudo /etc/init.d/dphys-swapfile restart || true
      echo "Temporary swap increased to 1500MB."
    fi
  fi
fi

echo "[3/6] Adding user '$USER' to hardware groups (video, gpio, dialout)..."
sudo usermod -a -G video,gpio,dialout "$USER" || true

echo "[4/6] Creating Python virtual environment..."
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  # Use --system-site-packages so system RPi.GPIO and gpiozero are inherited
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[5/6] Upgrading pip and installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pyserial

echo "[6/6] Verifying installation..."
python -c "import ear, liveness, hardware; print('Core modules imported successfully!')"

echo "=========================================================="
echo " Installation Complete!"
echo "=========================================================="
echo "To activate your environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To enroll a face:"
echo "  python enroll.py --name <YourName>"
echo ""
echo "To start unlock with Raspberry Pi GPIO (Relay on BCM 18):"
echo "  python unlock.py --gpio"
echo ""
echo "To start with an Arduino connected over USB Serial:"
echo "  python unlock.py --serial-port /dev/ttyACM0"
echo "=========================================================="
