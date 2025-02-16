#!/bin/sh
# Install CDO if possible
#

OS=$(uname -s)
if ! command -v cdo; then
    echo "CDO is not installed."
    if [ "$OS" = "Darwin" ]; then
        echo "Installing CDO on macOS..."
        brew install cdo
    elif [ "$OS" = "Linux" ]; then
        # Check if Ubuntu or Debian-based system
        if [ -f /etc/os-release ]; then
            # shellcheck disable=SC1091
            . /etc/os-release
            if [ "$ID" = "ubuntu" ] || [ "$ID_LIKE" = "debian" ]; then
                echo "Installing CDO on Ubuntu/Debian..."
                sudo apt-get update && sudo apt-get install --no-install-recommends -y cdo
            else
                echo "Unsupported Linux distribution: $ID"
            fi
        fi
    else
        echo "Unsupported OS: $OS"
    fi
else
	echo "CDO is already installed."
fi
