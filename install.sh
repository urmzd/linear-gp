#!/usr/bin/env bash
set -euo pipefail

REPO="urmzd/linear-gp"
BINARY="lgp"

# Allow overriding version; default to latest
VERSION="${LGP_VERSION:-}"
INSTALL_DIR="${LGP_INSTALL_DIR:-/usr/local/bin}"

die() { echo "error: $*" >&2; exit 1; }

detect_target() {
  local os arch target

  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux)  os="unknown-linux-gnu" ;;
    Darwin) os="apple-darwin" ;;
    *)      die "unsupported OS: $os" ;;
  esac

  case "$arch" in
    x86_64|amd64)  arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *)             die "unsupported architecture: $arch" ;;
  esac

  target="${arch}-${os}"
  echo "$target"
}

get_latest_version() {
  local url="https://api.github.com/repos/${REPO}/releases/latest"
  local tag
  tag="$(curl -fsSL "$url" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//')"
  [ -n "$tag" ] || die "failed to fetch latest release"
  echo "$tag"
}

main() {
  local target version archive_name url tmpdir

  target="$(detect_target)"

  if [ -z "$VERSION" ]; then
    echo "fetching latest release..."
    version="$(get_latest_version)"
  else
    version="$VERSION"
    # Ensure version starts with 'v'
    [[ "$version" == v* ]] || version="v${version}"
  fi

  archive_name="${BINARY}-${target}.tar.gz"
  url="https://github.com/${REPO}/releases/download/${version}/${archive_name}"

  echo "downloading ${BINARY} ${version} for ${target}..."
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  if ! curl -fsSL "$url" -o "${tmpdir}/${archive_name}"; then
    die "download failed. Check that ${version} has a prebuilt binary for ${target}.
Available releases: https://github.com/${REPO}/releases"
  fi

  echo "extracting..."
  tar -xzf "${tmpdir}/${archive_name}" -C "$tmpdir"

  echo "installing to ${INSTALL_DIR}..."
  if [ -w "$INSTALL_DIR" ]; then
    mv "${tmpdir}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
  else
    sudo mv "${tmpdir}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
  fi
  chmod +x "${INSTALL_DIR}/${BINARY}"

  echo "installed ${BINARY} ${version} to ${INSTALL_DIR}/${BINARY}"
}

main
