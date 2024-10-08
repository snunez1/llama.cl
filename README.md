<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

## llama.cl

This is a Common Lisp port of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) to idiomatic Common Lisp.

Why? Two reasons:

- Because Common Lisp is a fantastic language for experimentation, and this makes it easy to explore LLM techniques
- To serve as a reference implementation for the Common Lisp community

## How to run from emacs/slime/sly

### Prerequisites

We assume you have a working emacs, lisp and slime/sly setup.  Most of the systems `llama` requires are in [quicklisp](https://www.quicklisp.org/beta/), however Quicklisp isn't in the greatest of health, and the systems haven't been updated since June 2023.  Therefore you'll need to get at least [binary-types](https://github.com/snunez1/binary-types) from the repository, and [LLA](https://github.com/Lisp-Stat/lla) if you want to use BLAS/LAPACK libraries for matrix multiplication.  Put them in a location accessible to Quicklisp, like `~/common-lisp`.

1. Get the models from Karpathy's repo [(original instructions](https://github.com/karpathy/llama2.c#feel-the-magic)) pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) the dataset.

    ```bash
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
    ```
2. Load the file `run.lisp` into an emacs buffer
3. Load slime with `M-x slime`
4. Load LLA with `(ql:quickload :lla)` (optional - requires setup)
5. Load LLAMA with `(ql:quickload :llama)` from the REPL
6. Move into the package `(in-package :llama)`
7. Initalise the system with `(init #P"stories15M.bin" #P"tokenizer.bin" 32000)` (adjust paths if neccessary)
8. Generate a story with: `(generate *model* *tokenizer*)`

You can experiment with temperature, prompts and various samplers.  See code for all the options.

## Performance

My machine is running a 3.5 GHz 6-core Intel i7 5930, 256K/15MB cache with 64GB DDR4 RAM and with the `stories15M` I get about 2.5 tok/sec with CCL and 3.7 tok/s with SBCL.

If you want to use BLAS for matrix multiplication, you'll get about a 10X speed up.  Make sure that LLA is loaded _before_ you load `LLAMA`, if you do so it will automatically use the BLAS library.

Using LLA, the numbers are 14.4 tok/sec for CCL and 34.4 tok/sec for SBCL.

## Usage notes

dynamic variable binding
lisp heap size
etc


## Original README.md

For instructions on conversions to/from .bin format, training and other background, see the [original repo](https://github.com/karpathy/llama2.c)

