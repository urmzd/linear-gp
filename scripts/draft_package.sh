#!/bin/bash
set -e

# Configuration
SOURCE_DIR="docs"
BUILD_DIR="arxiv_build"
OUTPUT_FILE="arxiv_submission.tar.gz"
MAIN_TEX="thesis.tex"

echo "Cleaning up previous build..."
rm -rf "$BUILD_DIR" $OUTPUT_FILE

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"

echo "Copying source files..."
cp "$SOURCE_DIR"/*.tex "$BUILD_DIR/"
cp "$SOURCE_DIR"/*.bib "$BUILD_DIR/"
cp "$SOURCE_DIR"/*.cls "$BUILD_DIR/"

# Copy assets directly into build directory
echo "Copying assets..."
cp -r "assets" "$BUILD_DIR/"

cd "$BUILD_DIR"

echo "Updating asset paths in $MAIN_TEX..."
# Replace ../assets with assets to match local structure
sed -i.bak 's|\.\./assets|assets|g' "$MAIN_TEX"
rm "${MAIN_TEX}.bak"

echo "Modifying $MAIN_TEX for minted caching..."
# Switch from cache=false to finalizecache
# Use regex to be robust against whitespace or existing options
sed -i.bak 's/\\usepackage\[.*\]{minted}/\\usepackage\[finalizecache,cachedir=minted-cache\]{minted}/g' "$MAIN_TEX"
rm "${MAIN_TEX}.bak"

echo "Compiling to generate minted cache..."
if ! command -v pygmentize &> /dev/null; then
    echo "pygmentize not found. Setting up temporary virtualenv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install Pygments
fi

# Ensure Minted cache dir exists and is clean
rm -rf minted-cache
mkdir -p minted-cache

# First run with finalizecache to generate proper cache
pdflatex -shell-escape -interaction=nonstopmode "$MAIN_TEX" || true
bibtex "${MAIN_TEX%.*}" || true
pdflatex -shell-escape -interaction=nonstopmode "$MAIN_TEX" || true

echo "Switching to frozencache..."
sed -i.bak 's/\\usepackage\[finalizecache,cachedir=minted-cache\]{minted}/\\usepackage\[frozencache,cachedir=minted-cache\]{minted}/g' "$MAIN_TEX"
rm "${MAIN_TEX}.bak"

echo "Verifying compilation without shell-escape..."
pdflatex -interaction=nonstopmode "$MAIN_TEX" || true


# Create clean staging directory (use absolute path for reliability)
DRAFT_PKG_DIR="$(cd .. && pwd)/draft_packages"
rm -rf "$DRAFT_PKG_DIR"
mkdir -p "$DRAFT_PKG_DIR"

echo "Populating clean submission package in $DRAFT_PKG_DIR..."
# Copy only strictly required files
cp "$MAIN_TEX" "$DRAFT_PKG_DIR/"
cp "dalcsthesis.cls" "$DRAFT_PKG_DIR/"
cp "${MAIN_TEX%.*}.bbl" "$DRAFT_PKG_DIR/"
cp -r "minted-cache" "$DRAFT_PKG_DIR/"
cp -r "assets" "$DRAFT_PKG_DIR/"

echo "Testing compilation in draft_packages (without shell-escape)..."
pushd "$DRAFT_PKG_DIR" > /dev/null
pdflatex -interaction=nonstopmode "$MAIN_TEX" || true
pdflatex -interaction=nonstopmode "$MAIN_TEX" || true

echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.lof *.lot *.loa *.lol *.blg *.pdf
popd > /dev/null

echo "Creating archive from clean package..."
# Create archive from the staging directory contents
OUTPUT_PATH="$(cd .. && pwd)/$OUTPUT_FILE"
tar -czvf "$OUTPUT_PATH" -C "$DRAFT_PKG_DIR" .

echo "Done. Submission package at $OUTPUT_PATH"
echo "Clean content staged at $DRAFT_PKG_DIR"
