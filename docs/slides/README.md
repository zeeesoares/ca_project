# LaTeX Beamer Slides Template

This repository provides a modern and structured template for creating presentations using **LaTeX Beamer**. It is designed to streamline your workflow by isolating build artifacts and offering an automated compilation process via `Makefile`.

![Header Image](assets/dog.png)

## Prerequisites

To compile this template, you need a LaTeX distribution (TeX Live, MiKTeX, or MacTeX). On Ubuntu/Debian, you can install all necessary dependencies by running:

```bash
make install
```

# How to use

1. Compilation
```bash
make
```

2. Cleanup
```bash
make clean
```


# Project Structure

```bash
.
├── assets
│   └── dog.png
├── main.tex
├── Makefile
├── output
│   ├── main.log
│   ├── main.out
│   ├── main.pdf
│   └── ...
└── README.md

```