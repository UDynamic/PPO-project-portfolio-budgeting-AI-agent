New-Item -ItemType Directory -Force -Path "latexBuild" | Out-Null
Remove-Item -Path "latexBuild\*" -Force -ErrorAction SilentlyContinue

xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main
bibtex8 -W -c cp1256fa latexBuild\main
xindy -L persian-variant3 -C utf8 -I xindy -M latexBuild\main.xdy -t latexBuild\main.glg -o latexBuild\main.gls latexBuild\main.glo
xindy -L persian-variant3 -C utf8 -I xindy -M latexBuild\main.xdy -t latexBuild\main.blg -o latexBuild\main.bls latexBuild\main.blo
xindy -L english -C utf8 -I xindy -M latexBuild\main.xdy -t latexBuild\main.alg -o latexBuild\main.acr latexBuild\main.acn
makeindex latexBuild\main
xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main
xelatex -synctex=1 -interaction=nonstopmode -output-directory=latexBuild main