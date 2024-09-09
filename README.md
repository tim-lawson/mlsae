# Residual Stream Analysis with Multi-Layer SAEs

> [!NOTE]
> This repository accompanies the preprint Residual Stream Analysis with
> Multi-Layer SAEs: <https://arxiv.org/abs/2409.04185>. See
> [References](#references) for related work.

## Pretrained MLSAEs

We define two types of model: plain PyTorch
[MLSAE](./mlsae/model/autoencoder.py) modules, which are relatively small; and
PyTorch Lightning [MLSAETransformer](./mlsae/model/lightning.py) modules, which
include the underlying transformer. HuggingFace collections for both are here:

- [Multi-Layer Sparse Autoencoders](https://huggingface.co/collections/tim-lawson/multi-layer-sparse-autoencoders-66c2fe8896583c59b02ceb72)
- [Multi-Layer Sparse Autoencoders with Transformers](https://huggingface.co/collections/tim-lawson/multi-layer-sparse-autoencoders-with-transformers-66c441c87d1b24912175ce08)

We assume that pretrained MLSAEs have repo_ids with
[this naming convention](./mlsae/utils.py):

- `tim-lawson/mlsae-pythia-70m-deduped-x{expansion_factor}-k{k}`
- `tim-lawson/mlsae-pythia-70m-deduped-x{expansion_factor}-k{k}-tfm`

The Weights & Biases project for the paper is
[here](https://wandb.ai/timlawson-/mlsae).

## Installation

Install Python dependencies with Poetry:

```bash
poetry env use 3.12
poetry install
```

Install Python dependencies with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Node.js dependencies:

```bash
cd app
npm install
```

## Training

Train a single MLSAE:

```bash
python train.py --help
python train.py --model_name EleutherAI/pythia-70m-deduped --expansion_factor 64 -k 32
```

## Analysis

Test multiple pretrained MLSAEs:

```bash
python test.py --help
python test.py --model_name EleutherAI/pythia-70m-deduped --expansion_factor 32 64 128 -k 16 32 64
```

Compute the distributions of latent activations over layers for a single
pretrained MLSAE
([HuggingFace datasets](https://huggingface.co/collections/tim-lawson/mlsae-latent-distributions-over-layers-66d6a0ec9fcb6b494fb1808e)):

```bash
python -m mlsae.analysis.dists --help
python -m mlsae.analysis.dists --repo_id tim-lawson/mlsae-pythia-70m-deduped-x64-k32-tfm --max_tokens 100_000_000
```

Compute the maximally activating examples for each combination of latent and
layer for a single pretrained MLSAE
([HuggingFace datasets](https://huggingface.co/collections/tim-lawson/mlsae-maximally-activating-examples-66dbcc999a962ae594f631b6)):

```bash
python -m mlsae.analysis.examples --help
python -m mlsae.analysis.examples --repo_id tim-lawson/mlsae-pythia-70m-deduped-x64-k32-tfm --max_tokens 1_000_000
```

## Interactive visualizations

Run the interactive web application for a single pretrained MLSAE:

```bash
python -m mlsae.api --help
python -m mlsae.api --repo_id tim-lawson/mlsae-pythia-70m-deduped-x64-k32-tfm

cd app
npm run dev
```

Navigate to <http://localhost:3000>, enter a prompt, and click 'Submit'.

Alternatively, navigate to <http://localhost:3000/prompt/foobar>.

## Figures

Compute the mean cosine similarities between residual stream activation vectors
at adjacent layers of a single pretrained transformer:

```bash
python figures/resid_cos_sim.py --help
python figures/resid_cos_sim.py --model_name EleutherAI/pythia-70m-deduped
```

Save heatmaps of the distributions of latent activations over layers for
multiple pretrained MLSAEs:

```bash
python figures/dists_heatmaps.py --help
python figures/dists_heatmaps.py --expansion_factor 32 64 128 -k 16 32 64
```

Save a CSV of the mean standard deviations of the distributions of latent
activations over layers for multiple pretrained MLSAEs:

```bash
python figures/dists_layer_std.py --help
python figures/dists_layer_std.py --expansion_factor 32 64 128 -k 16 32 64
```

Save heatmaps of the maximum latent activations for a given prompt and multiple
pretrained MLSAEs:

```bash
python figures/prompt_heatmaps.py --help
python figures/prompt_heatmaps.py --expansion_factor 32 64 128 -k 16 32 64
```

Save a CSV of the Mean Max Cosine Similarity (MMCS) for multiple pretrained
MLSAEs:

```bash
python figures/mmcs.py --help
python figures/mmcs.py --expansion_factor 32 64 128 -k 16 32 64
```

## References

### Code

- <https://github.com/openai/sparse_autoencoder>
- <https://github.com/EleutherAI/sae>
- <https://github.com/ai-safety-foundation/sparse_autoencoder>
- <https://github.com/callummcdougall/sae_vis>

### Papers

- Gao et al. [2024] <https://cdn.openai.com/papers/sparse-autoencoders.pdf>
- Bricken et al. [2023]
  <https://transformer-circuits.pub/2023/monosemantic-features/index.html>
