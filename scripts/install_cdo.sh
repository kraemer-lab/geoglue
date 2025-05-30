#!/bin/bash

OS="$(uname -s)"

install_if_absent() {
    cmd="$1"
    pkgname_macos="${2:-$1}"
    pkgname_debian="${3:-$1}"
    if ! command -v "$cmd" > /dev/null; then
        echo "$1 is not installed."
        if [ "$OS" = "Darwin" ]; then
            echo "Installing $1 on macOS..."
            brew install "$pkgname_macos"
        elif [ "$OS" = "Linux" ]; then
            # Check if Ubuntu or Debian-based system
            if [ -f /etc/os-release ]; then
                # shellcheck disable=SC1091
                . /etc/os-release
                if [ "$ID" = "ubuntu" ] || [ "$ID_LIKE" = "debian" ]; then
                    echo "Installing CDO on Ubuntu/Debian..."
                    sudo apt-get update && sudo apt-get install --no-install-recommends -y "$pkgname_debian"
                else
                    echo "Unsupported Linux distribution: $ID"
                fi
            fi
        else
            echo "Unsupported OS: $OS"
        fi
    else
        echo "installed $(command -v "$cmd")"
    fi
}

install_if_absent cdo
