# A quick build process, without all crossrefs etc
pdflatex -output-directory=build writeup.tex
xdg-open build/writeup.pdf