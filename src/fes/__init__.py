"""Free-Energy Signature (FES) core: thermodynamic + RMT functionals over attention Laplacians."""

from .thermodynamics import (
    partition_Z,
    log_partition,
    free_energy_F,
    spectral_entropy_S,
    heat_capacity_C,
    spectral_form_factor_g,
    unfold_spectrum,
)
from .laplacian import (
    symmetrize_attention,
    combinatorial_laplacian,
    normalized_laplacian,
    attention_to_laplacian,
)
from .rmt_reference import goe_sff, deviation_from_goe
from .descriptors import build_layer_descriptor, build_fes

__all__ = [
    "partition_Z",
    "log_partition",
    "free_energy_F",
    "spectral_entropy_S",
    "heat_capacity_C",
    "spectral_form_factor_g",
    "unfold_spectrum",
    "symmetrize_attention",
    "combinatorial_laplacian",
    "normalized_laplacian",
    "attention_to_laplacian",
    "goe_sff",
    "deviation_from_goe",
    "build_layer_descriptor",
    "build_fes",
]
