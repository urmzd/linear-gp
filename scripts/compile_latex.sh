#!/usr/bin/env bash

#!/bin/bash

# Remove the file extension
filename=$(basename -- "$1")
filename="${filename%.*}"

# Compile using latexmk, which will handle multiple runs of pdflatex, bibtex, etc.
# Make sure to have the 'latexmk' and 'pygmentize' commands installed on your system.
latexmk -pdf -pdflatex="pdflatex -shell-escape -interaction=nonstopmode" -use-make "$filename.tex"

# Clean up auxiliary files
latexmk -c "$filename.tex"
