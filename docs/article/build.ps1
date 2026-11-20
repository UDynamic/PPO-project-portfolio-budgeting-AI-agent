Remove-Item -Path "latexBuild\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "main.aux","main.bbl","main.blg" -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "latexBuild" | Out-Null
New-Item -ItemType Directory -Force -Path "latexBuild\tex" | Out-Null

xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main
bibtex8 -W -c cp1256fa latexBuild\main
makeindex latexBuild\main
xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main
xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main

xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild highlights

xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild titlepage

xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main_anonymous
bibtex8 -W -c cp1256fa latexBuild\main_anonymous
makeindex latexBuild\main_anonymous
xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main_anonymous
xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main_anonymous