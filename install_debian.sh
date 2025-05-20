#!/bin/bash
set -e

# ---- Config ----
PYTHON_VERSION=3.12.3
ORIGINAL_DIR=$(pwd)

# ---- Functions ----

install_poetry() {
    if ! command -v poetry >/dev/null 2>&1; then
        echo "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        fi
    else
        echo "Poetry already installed: $(poetry --version)"
    fi
}

check_python12() {
    if command -v python3.12 >/dev/null 2>&1; then
        echo "Python 3.12 already installed: $(python3.12 --version)"
        return 0
    else
        return 1
    fi
}

install_python12() {
    echo "Installing Python $PYTHON_VERSION ..."
    sudo apt update
    sudo apt install -y wget build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev \
        xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

    cd /tmp
    wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
    tar xf Python-${PYTHON_VERSION}.tgz
    cd Python-${PYTHON_VERSION}

    ./configure --enable-optimizations
    make -j "$(nproc)"
    sudo make altinstall

    cd "$ORIGINAL_DIR"

    if command -v python3.12 >/dev/null 2>&1; then
        echo "Python 3.12 installation successful: $(python3.12 --version)"
    else
        echo "Python 3.12 installation failed."
        exit 1
    fi
}

# ---- Main Logic ----

# 1. Python
if ! check_python12; then
    install_python12
fi

# 2. Poetry
install_poetry

# 3. Project install steps
cd "$ORIGINAL_DIR"
poetry install

if [[ -x download_language_data.sh ]]; then
    sh download_language_data.sh
else
    echo "download_language_data.sh not found or not executable."
fi

