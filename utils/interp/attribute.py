"""Helper functions for visualising logit attribution"""

import plotly.express as px
from torch import Tensor
import einops
from jaxtyping import Float, Int

def get_logit_diff_directions(
    model,
    labels: Int[Tensor,"batch"]
) -> Float[Tensor,"d_model batch"]:
    return model.W_U[:, labels] - model.W_U[:, ~labels]

def get_logit_attribution(
    component_results: Float[Tensor,"... batch n_vertices d_model"],
    logit_diff_directions: Float[Tensor,"d_model batch"] 
) -> Float[Tensor,"..."]:

    batch_size = logit_diff_directions.size(-1)
    pooled_results: Float[Tensor,"... batch d_model"] = component_results.mean(dim=-2)
    mean_logit_attribution = einops.einsum(
        pooled_results,
        logit_diff_directions,
        "... batch d_model, d_model batch -> ..."
    )/batch_size
    return mean_logit_attribution

def plot_logit_attribution(logit_attribution: Float[Tensor,"n_components"],component_captions: list):
    px.line(
        x=component_captions,
        y=logit_attribution.detach(),
        labels={"x": "Component", "y": "Logit Attribution"},
        title=f"Component Logit Attributions"
    ).show()