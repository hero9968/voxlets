# A quick build process, without all crossrefs etc
latexdiff cvpr_writeup.tex writeup.tex > diff.tex
pdflatex -output-directory=build diff.tex
open build/diff.pdf