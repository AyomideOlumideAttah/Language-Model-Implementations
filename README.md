# Language Model Implementations

**Exploring the foundations of language generation — one word at a time.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AyomideOlumideAttah/Language-Model-Implementations/blob/main/language_model_implementations.ipynb)
![Python](https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![NLP](https://img.shields.io/badge/NLP-Statistical%20Language%20Models-purple)

## Overview

How can a machine learn to predict the next word using nothing more than the patterns in a corpus?

This project explores that question through word-level **n-gram language models implemented from scratch in Python**. It connects probability theory with practical natural language processing (NLP), from vocabulary construction and frequency estimation to probabilistic decoding and incremental text generation.

The notebook progresses from small, controlled examples to experiments with English Wikipedia, illustrating how context length and sampling strategies influence generated text. No pretrained neural language model is required.

## Features

- **Configurable n-gram order:** build unigram, bigram, trigram and higher-order models.
- **Vocabulary encoding:** map words to integer IDs and construct conditional distributions.
- **Linear interpolation:** combine predictions from multiple context lengths.
- **Four decoding strategies:** weighted random sampling, greedy decoding, top-k and top-p sampling.
- **Live generation:** display text progressively in a notebook.
- **Real-world corpus:** stream Wikipedia articles through Hugging Face Datasets and extract sentences with NLTK.

## How it works

An n-gram model approximates the probability of the next word using the preceding \(n-1\) words:

$$
P(w_t \mid w_1,\ldots,w_{t-1}) \approx P(w_t \mid w_{t-n+1},\ldots,w_{t-1})
$$

For a context \(c\), the maximum-likelihood estimate is

$$
P(w\mid c)=\frac{C(c,w)}{\sum_v C(c,v)}.
$$

Here, \(C(c,w)\) counts occurrences of a context followed by a particular word.

For a trigram model, linear interpolation aims to combine three distributions:

$$
P(w\mid u,v)=\lambda_3P_3(w\mid u,v)+\lambda_2P_2(w\mid v)+\lambda_1P_1(w)
$$

The weights should be nonnegative and sum to one. This is the intended mathematical formulation; see the implementation notes below for current normalization limitations.

### Model classes

| Class | Purpose |
| --- | --- |
| `NGramModel` | Builds vocabulary and probability tables for orders 1 through n. |
| `LIPNGramModel` | Extends the base model with three-component linear interpolation for sufficiently long contexts and orders of at least 3. |

### Decoding strategies

| Method | Behavior |
| --- | --- |
| `generate_random(context)` | Samples using the model's weights. |
| `generate_highest(context)` | Selects a highest-scoring word (greedy decoding). |
| `generate_top_k(context, k)` | Samples uniformly among up to k highest-scoring positive-weight candidates. |
| `generate_top_p(context, p)` | Samples by weight from a cumulative-score candidate set (intended nucleus sampling). |

## Getting started

### Option A — Google Colab (recommended)

1. Click the **Open In Colab** badge above.
2. Connect to a Python runtime. A CPU runtime is sufficient.
3. Add a code cell at the top and run:

   ```python
   %pip install -q numpy datasets nltk ipython
   ```

4. Run the two model-definition cells in order.
5. Run the small corpus examples before proceeding to the Wikipedia section.
6. Execute the Wikipedia cells in order if you want to explore the larger experiment. The notebook downloads NLTK's `punkt_tab` resource automatically.

An internet connection is required for package installation and Wikipedia streaming.

### Option B — Run locally with Jupyter

Clone the repository:

```bash
git clone https://github.com/AyomideOlumideAttah/Language-Model-Implementations.git
cd Language-Model-Implementations
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

Install dependencies and launch JupyterLab:

```bash
python -m pip install numpy datasets nltk ipython jupyterlab
jupyter lab language_model_implementations.ipynb
```

Select a Python kernel from the environment and execute the notebook from top to bottom. Start with the small examples if memory is limited.

## Quick example

After executing the model-definition cells, add the following cell:

```python
corpus = [
    "<s> I enjoy learning about language </s>",
    "<s> I enjoy learning about mathematics </s>",
    "<s> We enjoy studying language </s>",
]

n = 2
model = NGramModel(n, corpus)
context = ["<s>"]

for _ in range(20):
    word = model.generate_random(context)
    if not word or word == "</s>":
        break
    context.append(word)

print(" ".join(context[1:]))
```

An example of possible output is:

```text
I enjoy learning about language
```

Sampling is stochastic, so results may differ between runs. Start prompts with known vocabulary tokens, such as the sentence-start marker `<s>`.

## Experiments

| Corpus | Model order | Focus |
| --- | --- | --- |
| Fruit-themed toy corpus | 2 | Basic weighted generation |
| School-subject toy corpus | 3 | Top-k and top-p decoding |
| English Wikipedia sample | 5 | Generation with longer contexts and natural text |

The Wikipedia experiment uses the [Wikimedia Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia), configuration `20231101.en`. Articles are streamed, split into sentences and sampled until 500 eligible sentences are collected. This is a small exploratory corpus, not training on all of Wikipedia or a uniformly sampled global benchmark.

## Repository structure

```text
Language-Model-Implementations/
├── language_model_implementations.ipynb
└── README.md
```

The notebook contains the model implementations, corpus preparation and generation experiments.

## Implementation notes and limitations

- Higher-order tables use dense NumPy vectors per observed context. Memory can grow rapidly with vocabulary size; reduce the Wikipedia sample limit or model order if the runtime runs out of RAM.
- The current unigram implementation divides by vocabulary size rather than total token count. Weighted random sampling can still use these relative weights, but normalized probabilities are needed for perplexity and standard top-p decoding.
- The current interpolation routine references a global `n` in some branches; keep `n` consistent with the model order in the notebook, or replace those references with `self.n` before mixing model orders.
- Missing contexts can return zero vectors, and unseen prompt words may raise a lookup error. Interpolation does not currently guarantee normalization when component contexts are missing.
- The experiments demonstrate generation rather than held-out quantitative evaluation.

## Future directions

Possible extensions to the project include:

- Sparse successor counts and compact disk-backed storage.
- Properly normalized smoothing and unknown-word handling.
- Perplexity, coverage, runtime and memory benchmarks on held-out data.
- An interactive Gradio playground for comparing model orders and decoding methods.
- Comparisons with neural language models such as GPT-2.

## Author

**Ayomide Olumide-Attah** — [GitHub](https://github.com/AyomideOlumideAttah)

This project is an exploration of the statistical ideas underlying language modeling and a practical starting point for understanding more advanced NLP systems.

## References

- [Speech and Language Processing — Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/)
- [Wikimedia Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia)
- [NLTK documentation](https://www.nltk.org/)
- [NumPy documentation](https://numpy.org/doc/)
