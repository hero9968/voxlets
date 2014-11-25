# An example build process
# mkdir build
pdflatex -output-directory=build writeup.tex
bibtex -min-crossrefs=99 build/writeup.aux
pdflatex -output-directory=build writeup.tex
#pdflatex -output-directory=build writeup.tex