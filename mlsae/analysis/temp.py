import numpy as np
import torch


class StreamingMetric:
    def __init__(
        self, n_layers: int, n_latents: int, device: torch.device | str = "cpu"
    ) -> None:
        self.n_layers = n_layers
        self.n_latents = n_latents
        self.device = device

        # For E[Var[l|f]]
        self.sum_var_l_f = torch.zeros(n_latents, device=device)
        self.count_var_l_f = torch.zeros(n_latents, device=device)

        # For E[Var[l|t,f]]
        self.sum_var_l_tf = torch.zeros(n_latents, device=device)
        self.count_var_l_tf = torch.zeros(n_latents, device=device)

        # For Var[l]
        self.sum_l = torch.zeros(1, device=device)
        self.sum_l_squared = torch.zeros(1, device=device)
        self.count_l = 0

        self.layers = torch.arange(n_layers, device=device).float()

    def update(self, x: TopK) -> None:
        for layer in range(self.n_layers):
            indices = x.indices[layer].squeeze().view(-1)
            values = x.values[layer].squeeze().view(-1)

            # Compute probabilities for this token
            probs = torch.zeros(self.n_latents, device=self.device)
            probs.put_(indices, values, accumulate=True)

            # Update E[Var[l|f]]
            var_l_f = self.compute_var(probs)
            self.sum_var_l_f += var_l_f
            self.count_var_l_f += (probs > 0).float()

            # Update E[Var[l|t,f]]
            self.sum_var_l_tf += var_l_f
            self.count_var_l_tf += 1

            # Update Var[l]
            mean_layer = (self.layers * probs).sum()
            self.sum_l += mean_layer
            self.sum_l_squared += mean_layer**2
            self.count_l += 1

    def compute_var(self, probs: torch.Tensor) -> torch.Tensor:
        El = (self.layers * probs).sum()
        El2 = (self.layers**2 * probs).sum()
        return El2 - El**2

    def compute(self) -> dict[str, float]:
        # Compute E[Var[l|f]]
        E_Var_l_f = (self.sum_var_l_f / self.count_var_l_f.clamp(min=1)).mean().item()

        # Compute E[Var[l|t,f]]
        E_Var_l_tf = (
            (self.sum_var_l_tf / self.count_var_l_tf.clamp(min=1)).mean().item()
        )

        # Compute Var[l]
        mean_l = self.sum_l / self.count_l
        Var_l = (self.sum_l_squared / self.count_l - mean_l**2).item()

        # Compute fraction of variance explained
        frac_var_explained = E_Var_l_f / Var_l if Var_l != 0 else 0

        # Compute ratio of single token variance to aggregate variance
        single_token_ratio = E_Var_l_tf / E_Var_l_f if E_Var_l_f != 0 else 0

        return {
            "E_Var_l_f": E_Var_l_f,
            "E_Var_l_tf": E_Var_l_tf,
            "Var_l": Var_l,
            "frac_var_explained": frac_var_explained,
            "single_token_ratio": single_token_ratio,
        }


def get_stats(layer_std: torch.Tensor) -> dict[str, float]:
    values = layer_std.cpu().numpy()
    std = np.nanstd(values).item()
    return {
        "mean": np.nanmean(values).item(),
        "var": np.nanvar(values).item(),
        "std": std,
        "sem": std / np.sqrt(len(values)),
    }
