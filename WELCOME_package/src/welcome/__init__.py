"""WELCOME to the Universe: theory-driven full-sky weak-lensing realisations."""

from .core import (
    DEFAULT_DIAGNOSTIC_WAVELET_BANDS,
    WelcomeConfig,
    compute_theoretical_power_spectra,
    compute_theoretical_wavelet_l1,
    compute_theoretical_wavelet_l1_scales,
    generate_emulated_maps,
    run_welcome,
)

__version__ = "0.1.2"

__all__ = [
    "DEFAULT_DIAGNOSTIC_WAVELET_BANDS",
    "WelcomeConfig",
    "compute_theoretical_power_spectra",
    "compute_theoretical_wavelet_l1",
    "compute_theoretical_wavelet_l1_scales",
    "generate_emulated_maps",
    "run_welcome",
]
