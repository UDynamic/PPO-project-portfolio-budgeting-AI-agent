# Latexmk configuration for proper bibliography handling
$pdf_mode = 1;
$bibtex_use = 2;
$out_dir = 'latexBuild';

# Ensure bibtex only processes the main document
$bibtex = 'bibtex %O %S';

# Clean up auxiliary files
@default_files = ('main.tex');
