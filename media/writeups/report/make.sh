# An example build process
# mkdir build
pdflatex -output-directory=build survey.tex
bibtex -min-crossrefs=99 build/survey.aux
pdflatex -output-directory=build survey.tex
pdflatex -output-directory=build survey.tex