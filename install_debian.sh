curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc


# Set version to install
PYTHON_VERSION=3.12.3

ORIGINAL_DIR=$(pwd)

# Function to check if python3.12 is installed
check_python12() {
    if command -v python3.12 >/dev/null 2>&1; then
        echo "Python 3.12 is already installed: $(python3.12 --version)"
        exit 0
    else
        echo "Python 3.12 not found, proceeding to install..."
    fi
}

# Step 1: Check if already installed
check_python12

# Step 2: Install dependencies
sudo apt update
sudo apt install -y wget build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Step 3: Download and extract Python source
cd /tmp
wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar xf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

# Step 4: Configure, build, and install
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

# Step 5: Verify installation
if command -v python3.12 >/dev/null 2>&1; then
    echo "Python 3.12 installation successful: $(python3.12 --version)"
else
    echo "Python 3.12 installation failed."
    exit 1
fi


cd "$ORIGINAL_DIR"
 
poetry install 


