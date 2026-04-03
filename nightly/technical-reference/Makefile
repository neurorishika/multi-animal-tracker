.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -c
	rm -f main.bbl main.blg main.run.xml main.synctex.gz
