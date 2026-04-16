#!/bin/sh
# install.sh — Installs the lgp binary from GitHub releases.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/urmzd/linear-gp/main/install.sh | sh
#
# Environment variables:
#   LGP_VERSION     — version to install (e.g. "v1.7.5"); defaults to latest
#   LGP_INSTALL_DIR — installation directory; defaults to $HOME/.local/bin

set -eu

REPO="urmzd/linear-gp"
BINARY="lgp"

# curl with optional auth — uses GH_TOKEN or GITHUB_TOKEN if set.
gh_curl() {
    token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
    if [ -n "$token" ]; then
        curl -fsSL -H "Authorization: token $token" "$@"
    else
        curl -fsSL "$@"
    fi
}

main() {
    os=$(uname -s)
    arch=$(uname -m)

    case "$os" in
        Linux)
            case "$arch" in
                x86_64|amd64)  target="x86_64-unknown-linux-gnu" ;;
                arm64|aarch64) target="aarch64-unknown-linux-gnu" ;;
                *)             err "Unsupported Linux architecture: $arch" ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64|amd64)  target="x86_64-apple-darwin" ;;
                arm64|aarch64) target="aarch64-apple-darwin" ;;
                *)             err "Unsupported macOS architecture: $arch" ;;
            esac
            ;;
        *)
            err "Unsupported operating system: $os"
            ;;
    esac

    if [ -n "${LGP_VERSION:-}" ]; then
        tag="$LGP_VERSION"
        # Ensure version starts with 'v'
        case "$tag" in
            v*) ;;
            *)  tag="v${tag}" ;;
        esac
    else
        echo "Fetching latest release..."
        tag=$(gh_curl "https://api.github.com/repos/$REPO/releases/latest" \
            | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
        if [ -z "$tag" ]; then
            err "Failed to fetch latest release tag"
        fi
    fi

    archive_name="${BINARY}-${target}.tar.gz"
    url="https://github.com/$REPO/releases/download/${tag}/${archive_name}"

    install_dir="${LGP_INSTALL_DIR:-$HOME/.local/bin}"
    mkdir -p "$install_dir"

    echo "Downloading ${BINARY} ${tag} for ${target}..."
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT

    if ! gh_curl "$url" -o "${tmpdir}/${archive_name}"; then
        err "Download failed. Check that ${tag} has a prebuilt binary for ${target}.
Available releases: https://github.com/${REPO}/releases"
    fi

    echo "Extracting..."
    tar -xzf "${tmpdir}/${archive_name}" -C "$tmpdir"

    mv "${tmpdir}/${BINARY}" "${install_dir}/${BINARY}"
    chmod +x "${install_dir}/${BINARY}"

    echo "Installed ${BINARY} ${tag} to ${install_dir}/${BINARY}"

    case ":$PATH:" in
        *":$install_dir:"*) ;;
        *) add_to_path "$install_dir" ;;
    esac
}

add_to_path() {
    install_dir="$1"

    case "$(basename "$SHELL")" in
        zsh)  profile="$HOME/.zshrc" ;;
        bash)
            if [ -f "$HOME/.bashrc" ]; then
                profile="$HOME/.bashrc"
            else
                profile="$HOME/.profile"
            fi
            ;;
        fish) profile="$HOME/.config/fish/config.fish" ;;
        *)    profile="$HOME/.profile" ;;
    esac

    if [ "$(basename "$SHELL")" = "fish" ]; then
        if ! grep -q "$install_dir" "$profile" 2>/dev/null; then
            mkdir -p "$(dirname "$profile")"
            echo "" >> "$profile"
            echo "# Added by ${BINARY} installer" >> "$profile"
            echo "set -Ux fish_user_paths $install_dir \$fish_user_paths" >> "$profile"
            echo "Added $install_dir to $profile"
            echo "Restart your shell or run: source $profile"
        fi
    elif [ -n "$profile" ] && ! grep -q "$install_dir" "$profile" 2>/dev/null; then
        echo "" >> "$profile"
        echo "# Added by ${BINARY} installer" >> "$profile"
        echo "export PATH=\"$install_dir:\$PATH\"" >> "$profile"
        echo "Added $install_dir to $profile"
        echo "Restart your shell or run: source $profile"
    fi
}

err() {
    echo "Error: $1" >&2
    exit 1
}

main
