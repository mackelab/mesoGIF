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
# Install any bundled library dependencies
if [ -d "lib" ] && [ -n "$(ls "lib")" ]; then
    echo "You have bundled dependencies in the 'lib' directory."
    echo "This is intended for development only; when deploying, specify all dependencies in 'requirements.txt'"
    for package in $(ls lib); do
        pip install -e "lib/$package"
    done
fi
if [ -e "setup.py" ]; then  # Only install if there is project-specific code
    echo "Installing project code..."
    pip install -e .
fi
jupyter labextension install @jupyterlab/toc

echo "Registering IPython kernel for use in Jupyter."
# Get containing directory name, not whole path  (so "/home/alex/research/scripts/project" -> "project")
DIRNAME=${PWD##*/}
DISPLAYNAME="${DIRNAME//_/\ }"  # Replace underscores with spaces in display name
DISPLAYNAME="Python (""$DISPLAYNAME"")"  # Dislay kernel as "Python ([kernel])"
KERNELNAME="${DIRNAME//\ /_}"   # Replace spaces with underscores in kernel name
KERNELNAME="$(echo "$DIRNAME" | tr '[:upper:]' [':lower:'])"  # Make kernel name lower case
python -m ipykernel install --user --name $KERNELNAME --display-name "$DISPLAYNAME"

echo "Deactivating virtual environment"
deactivate
