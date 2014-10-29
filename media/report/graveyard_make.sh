# An example build process
# mkdir build
pdflatex -output-directory=build writeup_graveyard.tex
bibtex -min-crossrefs=99 build/writeup_graveyard.aux
pdflatex -output-directory=build writeup_graveyard.tex
#pdflatex -output-directory=build writeup.tex