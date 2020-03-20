# Change to the scripts directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Switching to directory $DIR"
cd "$DIR"

# Make sure dedicated environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment under 'venv'"
    python3 -m venv venv
fi

# Activate dedicated environment
echo "Activating virtual environment"
source venv/bin/activate

# venv typically starts with a really old pip version
pip install --upgrade pip

# Install code and its dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt
echo "\n\n\nInstalling project code..."
pip install ./code

echo "Deactivating virtual environment"
deactivate
