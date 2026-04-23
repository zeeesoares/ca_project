TEX = pdflatex
FLAGS = -output-directory=$(OUTPUT_DIR) -shell-restricted -interaction=nonstopmode
OUTPUT_DIR = output
FILE_NAME = main

.PHONY: all clean install compile

all: check compile

compile : $(FILE_NAME).tex
	@$(TEX) $(FLAGS) $< > /dev/null 2>&1
	@$(TEX) $(FLAGS) $< > /dev/null 2>&1

check : $(FILE_NAME).tex
	@mkdir -p $(OUTPUT_DIR)
	@[ -f $(FILE_NAME).tex ] && echo "$(FILE_NAME).tex" || echo "File does not exist: $(FILE_NAME).tex"

install:
	sudo apt update -y
	sudo apt install -y           \
		texlive-xetex 			  \
		texlive-latex-recommended \
		ttf-mscorefonts-installer \
		texlive-fonts-extra 	  \
		texlive-publishers 		  \
		fonts-stix

clean:
	@rm -rf \
		$(OUTPUT_DIR)/*.log 		\
		$(OUTPUT_DIR)/*.aux 		\
		$(OUTPUT_DIR)/*.lof 		\
		$(OUTPUT_DIR)/*.tof 		\
		$(OUTPUT_DIR)/*.tot 		\
		$(OUTPUT_DIR)/*.tol 		\
		$(OUTPUT_DIR)/*.toc 		\
		$(OUTPUT_DIR)/*.out 		\
		$(OUTPUT_DIR)/*.ilg 		\
		$(OUTPUT_DIR)/*.nls 		\
		$(OUTPUT_DIR)/*.nlo 		\
		$(OUTPUT_DIR)/*.fdb_latexmk \
		$(OUTPUT_DIR)/*.fls 		\
		$(OUTPUT_DIR)/*.nav 		\
		$(OUTPUT_DIR)/*.snm 