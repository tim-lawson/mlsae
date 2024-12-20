from typing import LiteralString


def model_card_template(transformer: bool) -> LiteralString:
    if transformer:
        return f"""{MODEL_CARD_TEMPLATE_START}

This model is a PyTorch Lightning MLSAETransformer module, which includes the underlying
transformer.

  {MODEL_CARD_TEMPLATE_END}"""
    return f"""{MODEL_CARD_TEMPLATE_START}

This model is a PyTorch TopKSAE module, which does not include the underlying
transformer.

  {MODEL_CARD_TEMPLATE_END}"""


MODEL_CARD_TEMPLATE_START = """
---
{{ card_data }}
---

# Model Card for {{ model_id }}

A Multi-Layer Sparse Autoencoder (MLSAE) trained on the residual stream activation
vectors from [{{ model_name }}](https://huggingface.co/{{ model_name }}) with an
expansion factor of R = {{ expansion_factor }} and sparsity k = {{ k }}, over 1 billion
tokens from [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted).
"""


MODEL_CARD_TEMPLATE_END = """
### Model Sources

- **Repository:** <https://github.com/tim-lawson/mlsae>
- **Paper:** <https://arxiv.org/abs/2409.04185>
- **Weights & Biases:** <https://wandb.ai/timlawson-/mlsae>

## Citation

**BibTeX:**

```bibtex
@misc{lawson_residual_2024,
  title         = {Residual {{ "{{" }}Stream Analysis{{ "}}" }} with {{ "{{" }}Multi-Layer SAEs{{ "}}" }}},
  author        = {Lawson, Tim and Farnik, Lucy and Houghton, Conor and Aitchison, Laurence},
  year          = {2024},
  month         = oct,
  number        = {arXiv:2409.04185},
  eprint        = {2409.04185},
  primaryclass  = {cs},
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2409.04185},
  urldate       = {2024-10-08},
  archiveprefix = {arXiv}
}
```
"""  # noqa: E501
