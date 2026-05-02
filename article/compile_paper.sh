#!/bin/bash
# Compilation script for the LaTeX paper
# Requires: pdflatex, bibtex

echo "=== Compiling LaTeX Paper ==="
echo ""

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install TeX Live or MiKTeX."
    echo ""
    echo "On Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "On macOS: brew install --cask mactex"
    echo "On Windows: Install MiKTeX from https://miktex.org/"
    exit 1
fi

# Create output directory
mkdir -p latexBuild

# First pass
echo "Running pdflatex (1st pass)..."
pdflatex -output-directory=latexBuild -interaction=nonstopmode main.tex

# Run bibtex for bibliography
echo "Running bibtex..."
cd latexBuild
bibtex main
cd ..

# Second pass (resolve references)
echo "Running pdflatex (2nd pass)..."
pdflatex -output-directory=latexBuild -interaction=nonstopmode main.tex

# Third pass (finalize)
echo "Running pdflatex (3rd pass)..."
pdflatex -output-directory=latexBuild -interaction=nonstopmode main.tex

echo ""
echo "=== Compilation Complete ==="
echo "Output: latexBuild/main.pdf"
echo ""

# Check if PDF was created
if [ -f "latexBuild/main.pdf" ]; then
    echo "✓ PDF generated successfully!"
    ls -lh latexBuild/main.pdf
else
    echo "✗ PDF generation failed. Check latexBuild/main.log for errors."
    exit 1
fi
