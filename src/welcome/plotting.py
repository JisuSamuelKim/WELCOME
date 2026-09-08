"""Plotting and lightweight diagnostics for WELCOME outputs."""

from __future__ import annotations

from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.linewidth': 3.5,
    'xtick.major.width': 3.25,
    'ytick.major.width': 3.25,
    'xtick.minor.width': 4.0,
    'ytick.minor.width': 4.0,
    'figure.titlesize': 20,
    'legend.handlelength': 2,
    'legend.handleheight': 0.7,
    'legend.labelspacing': 0.5,
    'lines.linewidth': 3.0,
    'lines.markersize': 3,
    'axes.grid': False,
    'grid.alpha': 0.75,
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'grid.color': 'gray',
})

from .core import (
    DEFAULT_DIAGNOSTIC_WAVELET_BANDS,
    compute_theoretical_wavelet_l1_scales,
    load_nz_file,
    top_hat_beam,
)


def _map_paths(result):
    paths = [Path(item["map_path"]) for item in result["emulation_results"]]
    if not paths:
        raise ValueError("No emulated maps are available in the run result.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing emulated map: " + str(missing[0]))
    return paths


def plot_example_map(result, realisation: int = 0, xsize: int = 700):
    """Display one generated full-sky convergence realisation."""
    paths = _map_paths(result)
    if not 0 <= int(realisation) < len(paths):
        raise IndexError(f"realisation must be between 0 and {len(paths) - 1}.")
    field = np.load(paths[int(realisation)], mmap_mode="r", allow_pickle=False)
    hp.mollview(
        field,
        title=f"WELCOME convergence realisation {int(realisation)}",
        unit=r"$\kappa$",
        xsize=int(xsize),
    )
    plt.show()


def plot_source_redshift_distribution(result):
    """Plot the configured single source plane or source redshift distribution."""
    config = result["config"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    if bool(config.single_plane):
        z0 = float(config.source_redshift)
        ax.axvline(z0, linewidth=2.0, label=rf"Single source plane, $z_s={z0:g}$")
        pad = max(0.25, 0.25 * z0)
        ax.set_xlim(max(0.0, z0 - pad), z0 + pad)
        ax.set_ylabel("source weight")
    else:
        z, nz = load_nz_file(config.nz_file)
        area = np.trapezoid(nz, z)
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("The supplied n(z) has a non-positive integral.")
        nz_plot = nz / area
        ax.plot(z, nz_plot, linewidth=2.0, label=r"Input $n(z)$")
        ax.set_xlim(max(0.0, float(z[0])), float(z[-1]))
        ax.set_ylabel(r"normalised $n(z)$")

    ax.set_xlabel("Redshift, z")
    ax.set_title("Source redshift distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def measure_emulated_power_spectra(result, use_cache: bool = True):
    """Measure C_ell from every generated map using the matched realisation order."""
    config = result["config"]
    paths = _map_paths(result)
    nside = int(config.nside)
    lmax = min(int(config.lmax), 3 * nside - 1)
    cache_path = Path(result["map_directory"]) / "emulation_power_spectra.npz"

    if use_cache and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            cached_paths = np.asarray(data["map_paths"], dtype=str).tolist()
            if (
                int(data["nside"]) == nside
                and int(data["lmax"]) == lmax
                and cached_paths == [str(path) for path in paths]
            ):
                return {
                    "ells": np.asarray(data["ells"], dtype=np.int64),
                    "all_cls": np.asarray(data["all_cls"], dtype=np.float64),
                    "path": cache_path,
                }

    all_cls = []
    for index, path in enumerate(paths):
        field = np.load(path, mmap_mode="r", allow_pickle=False)
        cl = hp.anafast(field, lmax=lmax, iter=int(config.harmonic_iter), pol=False)
        all_cls.append(np.asarray(cl, dtype=np.float64))
        print(f"[POWER] measured {index + 1}/{len(paths)}: {path.name}")

    all_cls = np.asarray(all_cls, dtype=np.float64)
    ells = np.arange(lmax + 1, dtype=np.int64)
    np.savez_compressed(
        cache_path,
        ells=ells,
        all_cls=all_cls,
        map_paths=np.asarray([str(path) for path in paths], dtype=str),
        nside=np.int64(nside),
        lmax=np.int64(lmax),
    )
    return {"ells": ells, "all_cls": all_cls, "path": cache_path}


def plot_power_realisations(result, use_cache: bool = True):
    """Compare matched theory and emulation power spectra using the theory binning."""
    measured = measure_emulated_power_spectra(result, use_cache=use_cache)
    theory = result["power_product"]

    theory_ells = np.asarray(theory["ells"], dtype=np.int64)
    theory_cls = np.asarray(theory["all_cls"], dtype=np.float64)
    emu_ells = np.asarray(measured["ells"], dtype=np.int64)
    emu_cls = np.asarray(measured["all_cls"], dtype=np.float64)

    binning_matrix = np.asarray(theory["binning_matrix"], dtype=np.float64)
    bin_ell_min = np.asarray(theory["correlation_bin_ell_min"], dtype=np.int64)
    bin_ell_max = np.asarray(theory["correlation_bin_ell_max"], dtype=np.int64)
    bin_ell_eff = np.asarray(theory["correlation_bin_ell_eff"], dtype=np.float64)

    if binning_matrix.shape[1] != theory_ells.size:
        raise ValueError(
            "The power-spectrum binning matrix is incompatible with the theory ell grid."
        )

    emu_lmin = int(np.min(emu_ells))
    emu_lmax = int(np.max(emu_ells))
    keep = (bin_ell_min >= max(2, emu_lmin)) & (bin_ell_max <= emu_lmax)
    if not np.any(keep):
        raise ValueError(
            "No theory power-spectrum bins are fully contained in the emulation ell range."
        )

    B = binning_matrix[keep]
    ell_eff = bin_ell_eff[keep]
    ell_min = bin_ell_min[keep]
    ell_max = bin_ell_max[keep]

    theory_binned = theory_cls @ B.T

    emu_on_theory_grid = np.zeros(
        (emu_cls.shape[0], theory_ells.size), dtype=np.float64
    )
    pos = np.searchsorted(theory_ells, emu_ells)
    valid = pos < theory_ells.size
    valid_indices = np.nonzero(valid)[0]
    exact = theory_ells[pos[valid]] == emu_ells[valid]
    valid_indices = valid_indices[exact]
    emu_on_theory_grid[:, pos[valid_indices]] = emu_cls[:, valid_indices]
    emu_binned = emu_on_theory_grid @ B.T

    n = min(theory_binned.shape[0], emu_binned.shape[0])
    theory_binned = theory_binned[:n]
    emu_binned = emu_binned[:n]

    good = (
        np.all(np.isfinite(theory_binned), axis=0)
        & np.all(np.isfinite(emu_binned), axis=0)
        & np.all(theory_binned > 0.0, axis=0)
    )
    if not np.any(good):
        raise ValueError("No finite positive matched power-spectrum bins are available.")

    ell_eff = ell_eff[good]
    ell_min = ell_min[good]
    ell_max = ell_max[good]
    theory_binned = theory_binned[:, good]
    emu_binned = emu_binned[:, good]

    residuals = 100.0 * (emu_binned / theory_binned - 1.0)
    mean_residual = np.mean(residuals, axis=0)
    residual_low = np.percentile(residuals, 16.0, axis=0)
    residual_high = np.percentile(residuals, 84.0, axis=0)

    theory_mean = np.mean(theory_binned, axis=0)
    emu_mean = np.mean(emu_binned, axis=0)

    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    theory_colour = colours[0]
    emu_colour = colours[1 % len(colours)]

    for index in range(n):
        ax.loglog(
            ell_eff,
            theory_binned[index],
            linewidth=0.7,
            alpha=0.10,
            color=theory_colour,
        )
        ax.loglog(
            ell_eff,
            emu_binned[index],
            linewidth=0.7,
            alpha=0.10,
            color=emu_colour,
        )
        ax_res.semilogx(
            ell_eff,
            residuals[index],
            linewidth=0.6,
            alpha=0.10,
            color=emu_colour,
        )

    ax.loglog(
        ell_eff,
        theory_mean,
        linewidth=2.2,
        color=theory_colour,
        label="Theory mean",
    )
    ax.loglog(
        ell_eff,
        emu_mean,
        linewidth=2.2,
        linestyle="--",
        color=emu_colour,
        label="Emulation mean",
    )

    ax_res.fill_between(
        ell_eff,
        residual_low,
        residual_high,
        alpha=0.15,
        color=emu_colour,
        label="16-84% range",
    )
    ax_res.semilogx(
        ell_eff,
        mean_residual,
        linewidth=2.0,
        color=emu_colour,
        label="Mean residual",
    )
    ax_res.axhline(0.0, linewidth=1.0)

    ax.set_ylim(1.0e-11, 1.0e-7)
    ax_res.set_ylim(-50.0, 50.0)

    # Fill the horizontal plotting range from the first to the last
    # actual binned data point. The retained bins themselves are unchanged.
    x_min = float(ell_eff[0])
    x_max = float(ell_eff[-1])
    ax.set_xlim(x_min, x_max)
    ax_res.set_xlim(x_min, x_max)
    ax.margins(x=0.0)
    ax_res.margins(x=0.0)

    ax_res.set_yticks([-50.0, -25.0, 0.0, 25.0, 50.0])

    ax.set_ylabel(r"$C_\ell^{\kappa\kappa}$")
    ax.legend(frameon=False)

    ax_res.set_xlabel(r"Multipole $\ell$")
    ax_res.set_ylabel("Residual [%]")

    fig.tight_layout()
    return fig, (ax, ax_res)


def _wavelet_filters(bands, lmax):
    unique_scales = sorted({float(scale) for band in bands for scale in band})
    beams = {scale: np.asarray(top_hat_beam(scale, lmax), dtype=np.float64) for scale in unique_scales}
    return np.asarray([beams[float(a)] - beams[float(b)] for a, b in bands], dtype=np.float64)


def _wavelet_cache_path(result, bands):
    token = "_".join(f"{a:g}-{b:g}" for a, b in bands).replace(".", "p")
    return Path(result["map_directory"]) / f"wavelet_l1_diagnostics_{token}.npz"


def _prepared_wavelet_cache_path(result, bands):
    token = "_".join(f"{a:g}-{b:g}" for a, b in bands).replace(".", "p")
    return Path(result["map_directory"]) / f"wavelet_l1_prepared_{token}.npz"


def measure_emulated_wavelet_l1(
    result,
    bands=DEFAULT_DIAGNOSTIC_WAVELET_BANDS,
    n_bins: int = 400,
    use_cache: bool = True,
):
    """Measure wavelet L1 curves from all generated maps on common kappa grids."""
    bands = tuple((float(a), float(b)) for a, b in bands)
    config = result["config"]
    paths = _map_paths(result)
    nside = int(config.nside)
    lmax = min(int(config.lmax), 3 * nside - 1)
    cache_path = _wavelet_cache_path(result, bands)

    if use_cache and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            cached_bands = np.asarray(data["bands"], dtype=np.float64)
            cached_paths = np.asarray(data["map_paths"], dtype=str).tolist()
            if (
                np.array_equal(cached_bands, np.asarray(bands, dtype=np.float64))
                and int(data["nside"]) == nside
                and int(data["lmax"]) == lmax
                and int(data["n_bins"]) == int(n_bins)
                and cached_paths == [str(path) for path in paths]
            ):
                print(f"[WAVELET] Loaded cached 400-bin measurements: {cache_path}")
                return {
                    "bands": bands,
                    "kappa": np.asarray(data["kappa"], dtype=np.float64),
                    "all_l1": np.asarray(data["all_l1"], dtype=np.float64),
                    "path": cache_path,
                }

    filters = _wavelet_filters(bands, lmax)
    minima = np.full(len(bands), np.inf, dtype=np.float64)
    maxima = np.full(len(bands), -np.inf, dtype=np.float64)

    for index, path in enumerate(paths):
        field = np.load(path, mmap_mode="r", allow_pickle=False)
        alm = hp.map2alm(field, lmax=lmax, iter=int(config.harmonic_iter), pol=False)
        for scale_index, wavelet_filter in enumerate(filters):
            component = hp.alm2map(
                hp.almxfl(alm, wavelet_filter, inplace=False),
                nside=nside,
                lmax=lmax,
                pol=False,
            )
            finite = np.asarray(component[np.isfinite(component)], dtype=np.float64)
            if finite.size == 0:
                raise ValueError(f"No finite wavelet values for {bands[scale_index]}.")
            minima[scale_index] = min(minima[scale_index], float(np.min(finite)))
            maxima[scale_index] = max(maxima[scale_index], float(np.max(finite)))
        print(f"[WAVELET RANGE] measured {index + 1}/{len(paths)}: {path.name}")

    edges = np.empty((len(bands), int(n_bins) + 1), dtype=np.float64)
    for scale_index in range(len(bands)):
        span = maxima[scale_index] - minima[scale_index]
        pad = max(span * 1.0e-10, np.finfo(np.float64).eps)
        edges[scale_index] = np.linspace(
            minima[scale_index] - pad,
            maxima[scale_index] + pad,
            int(n_bins) + 1,
        )
    kappa = 0.5 * (edges[:, :-1] + edges[:, 1:])

    all_l1 = np.empty((len(paths), len(bands), int(n_bins)), dtype=np.float64)
    for index, path in enumerate(paths):
        field = np.load(path, mmap_mode="r", allow_pickle=False)
        alm = hp.map2alm(field, lmax=lmax, iter=int(config.harmonic_iter), pol=False)
        for scale_index, wavelet_filter in enumerate(filters):
            component = hp.alm2map(
                hp.almxfl(alm, wavelet_filter, inplace=False),
                nside=nside,
                lmax=lmax,
                pol=False,
            )
            values = np.asarray(component[np.isfinite(component)], dtype=np.float64)
            counts, _ = np.histogram(values, bins=edges[scale_index], density=False)
            widths = np.diff(edges[scale_index])
            pdf = counts.astype(np.float64) / (float(np.sum(counts)) * widths)
            area = np.trapezoid(pdf, kappa[scale_index])
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError(f"Invalid wavelet PDF for {bands[scale_index]}.")
            pdf = pdf / area
            all_l1[index, scale_index] = np.abs(kappa[scale_index]) * pdf
        print(f"[WAVELET L1] measured {index + 1}/{len(paths)}: {path.name}")

    np.savez_compressed(
        cache_path,
        bands=np.asarray(bands, dtype=np.float64),
        kappa=kappa,
        all_l1=all_l1,
        map_paths=np.asarray([str(path) for path in paths], dtype=str),
        nside=np.int64(nside),
        lmax=np.int64(lmax),
        n_bins=np.int64(n_bins),
    )
    return {"bands": bands, "kappa": kappa, "all_l1": all_l1, "path": cache_path}


def _load_prepared_wavelet_diagnostics(result, bands, n_bins):
    """Load complete theory-plus-emulation diagnostics when the cache matches."""
    cache_path = _prepared_wavelet_cache_path(result, bands)
    if not cache_path.is_file():
        return None

    config = result["config"]
    paths = _map_paths(result)
    nside = int(config.nside)
    lmax = min(int(config.lmax), 3 * nside - 1)

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_bands = np.asarray(data["bands"], dtype=np.float64)
            cached_paths = np.asarray(data["map_paths"], dtype=str).tolist()
            if not (
                np.array_equal(cached_bands, np.asarray(bands, dtype=np.float64))
                and int(data["nside"]) == nside
                and int(data["lmax"]) == lmax
                and int(data["n_bins"]) == int(n_bins)
                and cached_paths == [str(path) for path in paths]
            ):
                return None

            products = {}
            for scale_index, (theta1, theta2) in enumerate(bands):
                label = f"{theta1:g}-{theta2:g}"
                products[label] = {
                    "kappa": np.asarray(
                        data[f"theory_kappa_{scale_index}"], dtype=np.float64
                    ),
                    "l1": np.asarray(
                        data[f"theory_l1_{scale_index}"], dtype=np.float64
                    ),
                }
                pdf_key = f"theory_pdf_{scale_index}"
                if pdf_key in data.files:
                    products[label]["pdf"] = np.asarray(
                        data[pdf_key], dtype=np.float64
                    )

            theory_path = None
            if "theory_path" in data.files:
                saved_theory_path = str(np.asarray(data["theory_path"]).item())
                if saved_theory_path:
                    theory_path = Path(saved_theory_path)

            diagnostics = {
                "bands": bands,
                "theory": {
                    "bands": bands,
                    "products": products,
                    "path": theory_path,
                },
                "emulation": {
                    "bands": bands,
                    "kappa": np.asarray(data["emulation_kappa"], dtype=np.float64),
                    "all_l1": np.asarray(
                        data["emulation_all_l1"], dtype=np.float64
                    ),
                    "path": Path(str(np.asarray(data["emulation_path"]).item())),
                },
                "path": cache_path,
            }
    except (KeyError, ValueError, OSError):
        return None

    print(f"[WAVELET] Loaded complete cached diagnostics: {cache_path}")
    return diagnostics


def _save_prepared_wavelet_diagnostics(result, diagnostics, n_bins):
    """Save complete diagnostics so theory is not recomputed after restart."""
    bands = diagnostics["bands"]
    config = result["config"]
    paths = _map_paths(result)
    nside = int(config.nside)
    lmax = min(int(config.lmax), 3 * nside - 1)
    cache_path = _prepared_wavelet_cache_path(result, bands)

    payload = {
        "bands": np.asarray(bands, dtype=np.float64),
        "map_paths": np.asarray([str(path) for path in paths], dtype=str),
        "nside": np.int64(nside),
        "lmax": np.int64(lmax),
        "n_bins": np.int64(n_bins),
        "emulation_kappa": np.asarray(
            diagnostics["emulation"]["kappa"], dtype=np.float64
        ),
        "emulation_all_l1": np.asarray(
            diagnostics["emulation"]["all_l1"], dtype=np.float64
        ),
        "emulation_path": np.asarray(str(diagnostics["emulation"]["path"])),
        "theory_path": np.asarray(
            ""
            if diagnostics["theory"].get("path") is None
            else str(diagnostics["theory"]["path"])
        ),
    }

    theory_products = diagnostics["theory"]["products"]
    for scale_index, (theta1, theta2) in enumerate(bands):
        label = f"{theta1:g}-{theta2:g}"
        product = theory_products[label]
        payload[f"theory_kappa_{scale_index}"] = np.asarray(
            product["kappa"], dtype=np.float64
        )
        payload[f"theory_l1_{scale_index}"] = np.asarray(
            product["l1"], dtype=np.float64
        )
        if "pdf" in product:
            payload[f"theory_pdf_{scale_index}"] = np.asarray(
                product["pdf"], dtype=np.float64
            )

    np.savez_compressed(cache_path, **payload)
    return cache_path


def prepare_wavelet_diagnostics(
    result,
    bands=DEFAULT_DIAGNOSTIC_WAVELET_BANDS,
    n_bins: int = 400,
    use_cache: bool = True,
):
    """Prepare cached 400-bin theory-versus-emulation wavelet diagnostics."""
    bands = tuple((float(a), float(b)) for a, b in bands)

    if use_cache:
        cached = _load_prepared_wavelet_diagnostics(result, bands, n_bins)
        if cached is not None:
            return cached

    theory = compute_theoretical_wavelet_l1_scales(
        result["config"],
        bands=bands,
        save=True,
    )
    emulation = measure_emulated_wavelet_l1(
        result,
        bands=bands,
        n_bins=n_bins,
        use_cache=use_cache,
    )
    diagnostics = {
        "bands": bands,
        "theory": theory,
        "emulation": emulation,
    }
    diagnostics["path"] = _save_prepared_wavelet_diagnostics(
        result,
        diagnostics,
        n_bins,
    )
    print(f"[WAVELET] Saved complete diagnostics cache: {diagnostics['path']}")
    return diagnostics


def _finite_curve_domain(x, y):
    """Return the finite x-domain of a curve."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        raise ValueError("A wavelet L1 curve has fewer than two finite samples.")
    return float(np.min(x[good])), float(np.max(x[good]))


def _information_support(x, y, peak_fraction=0.01):
    """Return outer support bounds above a fraction of the curve peak."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        raise ValueError("A wavelet L1 curve has fewer than two finite samples.")

    x = x[good]
    y = np.clip(y[good], 0.0, None)
    peak = float(np.max(y))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("A wavelet L1 curve has no positive finite peak.")

    informative = y > float(peak_fraction) * peak
    if np.count_nonzero(informative) < 2:
        raise ValueError(
            "A wavelet L1 curve has insufficient support above the threshold."
        )
    return float(np.min(x[informative])), float(np.max(x[informative]))


def _interpolate_finite_curve(x, y, x_new):
    """Linearly interpolate a finite curve onto a common kappa grid."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, unique = np.unique(x, return_index=True)
    y = y[unique]
    if x.size < 2:
        raise ValueError(
            "A wavelet L1 curve has fewer than two unique kappa samples."
        )
    return np.interp(x_new, x, y)


def _theory_pdf_sigma(product):
    """Return the conventional standard deviation of the theoretical PDF."""
    if "pdf" not in product:
        raise KeyError(
            "The theoretical wavelet product does not contain 'pdf', "
            "which is required for sigma_theory."
        )

    x = np.asarray(product["kappa"], dtype=np.float64)
    pdf = np.asarray(product["pdf"], dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(pdf)
    x = x[good]
    pdf = np.clip(pdf[good], 0.0, None)

    order = np.argsort(x)
    x = x[order]
    pdf = pdf[order]

    norm = np.trapezoid(pdf, x)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("The theoretical wavelet PDF has a non-positive integral.")
    pdf = pdf / norm

    mean = np.trapezoid(x * pdf, x)
    variance = np.trapezoid((x - mean) ** 2 * pdf, x)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("The theoretical wavelet PDF has a non-positive variance.")
    return float(np.sqrt(variance))


def plot_wavelet_l1(diagnostics):
    """Plot theory/emulation wavelet L1 curves with 16-84% ensemble bands."""
    CENTRAL_FLOOR_FRACTION = 0.1

    bands = tuple(diagnostics["bands"])
    theory_products = diagnostics["theory"]["products"]
    emu_kappa = np.asarray(diagnostics["emulation"]["kappa"], dtype=np.float64)
    all_l1 = np.asarray(diagnostics["emulation"]["all_l1"], dtype=np.float64)

    if len(bands) != all_l1.shape[1]:
        raise ValueError(
            "Wavelet band metadata does not match the emulation diagnostics."
        )

    emu_mean = np.nanmean(all_l1, axis=0)
    emu_p16 = np.nanpercentile(all_l1, 16.0, axis=0)
    emu_p84 = np.nanpercentile(all_l1, 84.0, axis=0)

    def central_low_l1_mask(kappa, theory_l1, fraction):
        kappa = np.asarray(kappa, dtype=np.float64)
        theory_l1 = np.asarray(theory_l1, dtype=np.float64)

        floor = fraction * np.nanmax(theory_l1)
        centre = int(np.argmin(np.abs(kappa)))

        left = centre
        while (
            left > 0
            and np.isfinite(theory_l1[left])
            and theory_l1[left] <= floor
        ):
            left -= 1

        right = centre
        while (
            right < len(theory_l1) - 1
            and np.isfinite(theory_l1[right])
            and theory_l1[right] <= floor
        ):
            right += 1

        central_mask = np.zeros_like(theory_l1, dtype=bool)
        central_mask[left + 1:right] = True
        return central_mask, floor, left, right

    def interpolate_central_gap(x, y, central_mask):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        output = y.copy()

        missing = central_mask & np.isfinite(x)
        valid = np.isfinite(x) & np.isfinite(y) & ~central_mask

        if np.count_nonzero(valid) >= 2 and np.any(missing):
            x_valid = x[valid]
            y_valid = y[valid]
            order = np.argsort(x_valid)
            x_valid = x_valid[order]
            y_valid = y_valid[order]
            output[missing] = np.interp(
                x[missing],
                x_valid,
                y_valid,
            )

        return output

    # Common informative physical-kappa support across all eight curves.
    lower = []
    upper = []

    for scale_index, (theta1, theta2) in enumerate(bands):
        label = f"{theta1:g}-{theta2:g}"
        theory = theory_products[label]

        lo, hi = _information_support(
            theory["kappa"],
            theory["l1"],
            peak_fraction=0.01,
        )
        lower.append(lo)
        upper.append(hi)

        lo, hi = _information_support(
            emu_kappa[scale_index],
            emu_mean[scale_index],
            peak_fraction=0.01,
        )
        lower.append(lo)
        upper.append(hi)

    common_min = max(lower)
    common_max = min(upper)

    if (
        not np.isfinite(common_min)
        or not np.isfinite(common_max)
        or common_min >= common_max
    ):
        raise ValueError(
            "The eight wavelet L1 curves have no common informative kappa support."
        )

    common_kappa = np.linspace(
        common_min,
        common_max,
        400,
        dtype=np.float64,
    )

    n_scales = len(bands)
    n_kappa = common_kappa.size

    theory_common = np.empty((n_scales, n_kappa), dtype=np.float64)
    emu_mean_common = np.empty_like(theory_common)
    emu_p16_common = np.empty_like(theory_common)
    emu_p84_common = np.empty_like(theory_common)
    sigma_theory = np.empty(n_scales, dtype=np.float64)

    for scale_index, (theta1, theta2) in enumerate(bands):
        label = f"{theta1:g}-{theta2:g}"
        theory = theory_products[label]

        theory_common[scale_index] = _interpolate_finite_curve(
            theory["kappa"],
            theory["l1"],
            common_kappa,
        )
        emu_mean_common[scale_index] = _interpolate_finite_curve(
            emu_kappa[scale_index],
            emu_mean[scale_index],
            common_kappa,
        )
        emu_p16_common[scale_index] = _interpolate_finite_curve(
            emu_kappa[scale_index],
            emu_p16[scale_index],
            common_kappa,
        )
        emu_p84_common[scale_index] = _interpolate_finite_curve(
            emu_kappa[scale_index],
            emu_p84[scale_index],
            common_kappa,
        )
        sigma_theory[scale_index] = _theory_pdf_sigma(theory)

    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.5),
        sharex=False,
        gridspec_kw={
            "height_ratios": [3.0, 1.25],
            "hspace": 0.18,
        },
    )

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    scale_handles = []

    # Accuracy guide regions.
    ax_res.axhspan(-5.0, 5.0, color="0.90", zorder=0)
    ax_res.axhspan(-2.0, 2.0, color="0.80", zorder=0)

    for scale_index, (theta1, theta2) in enumerate(bands):
        colour = colours[scale_index % len(colours)]
        scale_label = f"{theta1:g}-{theta2:g}"

        theory_l1 = theory_common[scale_index]
        emu_l1 = emu_mean_common[scale_index]
        emu_l1_p16 = emu_p16_common[scale_index]
        emu_l1_p84 = emu_p84_common[scale_index]

        # Upper panel: 16-84% emulation band, theory, and emulation mean.
        ax.fill_between(
            common_kappa,
            emu_l1_p16,
            emu_l1_p84,
            color=colour,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            common_kappa,
            theory_l1,
            color=colour,
            linestyle="-",
            zorder=3,
        )
        ax.plot(
            common_kappa,
            emu_l1,
            color=colour,
            linestyle="--",
            zorder=4,
        )

        central_mask, floor, left, right = central_low_l1_mask(
            common_kappa,
            theory_l1,
            CENTRAL_FLOOR_FRACTION,
        )

        x_sigma = common_kappa / sigma_theory[scale_index]

        # Mean residual.
        valid_raw = (
            np.isfinite(x_sigma)
            & np.isfinite(theory_l1)
            & np.isfinite(emu_l1)
            & (theory_l1 > 0.0)
            & ~central_mask
        )

        residual_raw = np.full_like(theory_l1, np.nan)
        residual_raw[valid_raw] = 100.0 * (
            emu_l1[valid_raw] / theory_l1[valid_raw] - 1.0
        )

        # Residual 16-84% emulation band.
        band16_raw = np.full_like(theory_l1, np.nan)
        band84_raw = np.full_like(theory_l1, np.nan)

        valid_band = (
            np.isfinite(x_sigma)
            & np.isfinite(theory_l1)
            & np.isfinite(emu_l1_p16)
            & np.isfinite(emu_l1_p84)
            & (theory_l1 > 0.0)
            & ~central_mask
        )

        band16_raw[valid_band] = 100.0 * (
            emu_l1_p16[valid_band] / theory_l1[valid_band] - 1.0
        )
        band84_raw[valid_band] = 100.0 * (
            emu_l1_p84[valid_band] / theory_l1[valid_band] - 1.0
        )

        # Plot-only interpolation through the excluded central trough.
        residual_plot = interpolate_central_gap(
            x_sigma,
            residual_raw,
            central_mask,
        )
        band16_plot = interpolate_central_gap(
            x_sigma,
            band16_raw,
            central_mask,
        )
        band84_plot = interpolate_central_gap(
            x_sigma,
            band84_raw,
            central_mask,
        )

        # Visible residual range: +/- 2 sigma_theory.
        visible = (
            np.isfinite(x_sigma)
            & np.isfinite(residual_plot)
            & np.isfinite(band16_plot)
            & np.isfinite(band84_plot)
            & (x_sigma >= -2.0)
            & (x_sigma <= 2.0)
        )

        ax_res.fill_between(
            x_sigma[visible],
            band16_plot[visible],
            band84_plot[visible],
            color=colour,
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )
        ax_res.plot(
            x_sigma[visible],
            residual_plot[visible],
            color=colour,
            linestyle="--",
            zorder=3,
            clip_on=True,
        )

        scale_handles.append(
            Line2D(
                [0],
                [0],
                color=colour,
                linestyle="-",
                label=scale_label,
            )
        )

    scale_legend = ax.legend(
        handles=scale_handles,
        loc="upper left",
        frameon=False,
    )
    ax.add_artist(scale_legend)

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                label="Theory",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="Emulation",
            ),
        ],
        loc="upper right",
        frameon=False,
    )

    ax_res.axhline(
        0.0,
        color="black",
        zorder=1,
    )

    # Upper panel.
    ax.set_xlim(common_min, common_max)
    ax.margins(x=0)
    ax.set_ylabel(r"$\ell_1(\kappa)$")
    ax.set_xlabel(
        r"$\kappa$",
        fontsize=16,
        labelpad=4,
    )
    ax.tick_params(
        axis="x",
        labelsize=12,
        pad=2,
    )
    ax.tick_params(
        axis="y",
        labelsize=12,
        pad=2,
    )

    # Residual panel.
    ax_res.set_xlim(-2.0, 2.0)
    ax_res.set_ylim(-10.0, 10.0)
    ax_res.set_xticks(
        [
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
        ]
    )
    ax_res.tick_params(
        axis="x",
        labelsize=12,
        pad=2,
    )
    ax_res.tick_params(
        axis="y",
        labelsize=12,
        pad=2,
    )
    ax_res.set_yticks([-10, -5, 0, 5, 10])
    ax_res.set_xlabel(
        r"$\kappa/\sigma_{\rm theory}$",
        fontsize=16,
        labelpad=3,
    )
    ax_res.set_ylabel("Residual [%]")
    ax_res.margins(x=0)

    plt.show()
    return fig, (ax, ax_res)

def l1_weighted_moments(kappa, l1_curve):
    """Return mean, variance, skewness and excess kurtosis of a unit-area L1 curve."""
    x = np.asarray(kappa, dtype=np.float64)
    y = np.asarray(l1_curve, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = np.clip(y[good], 0.0, None)
    if x.size < 2:
        return np.full(4, np.nan, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    norm = np.trapezoid(y, x)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.full(4, np.nan, dtype=np.float64)
    q = y / norm
    mean = np.trapezoid(q * x, x)
    delta = x - mean
    mu2 = np.trapezoid(q * delta**2, x)
    mu3 = np.trapezoid(q * delta**3, x)
    mu4 = np.trapezoid(q * delta**4, x)
    if not np.isfinite(mu2) or mu2 <= 0.0:
        return np.asarray([mean, mu2, np.nan, np.nan], dtype=np.float64)
    sigma = np.sqrt(mu2)
    return np.asarray([mean, mu2, mu3 / sigma**3, mu4 / mu2**2 - 3.0], dtype=np.float64)


def _moment_informative_support(x, y, fraction=0.01):
    """Return the outer support where an L1 curve exceeds a fraction of its peak."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = np.clip(y[valid], 0.0, None)
    if x.size < 2:
        raise ValueError("Too few finite points in the L1 curve.")
    peak = float(np.nanmax(y))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("Invalid L1 curve peak.")
    keep = y > float(fraction) * peak
    if np.count_nonzero(keep) < 2:
        raise ValueError("Informative L1 support is too small.")
    return float(np.min(x[keep])), float(np.max(x[keep]))


def _moment_interp_curve(x, y, x_new):
    """Interpolate a finite one-dimensional curve onto a new grid."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        raise ValueError("Too few finite points for interpolation.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, unique = np.unique(x, return_index=True)
    y = y[unique]
    if x.size < 2:
        raise ValueError("Too few unique points for interpolation.")
    return np.interp(x_new, x, y)


def plot_wavelet_moments(diagnostics):
    """Compare theory with emulation L1-weighted moments on identical support and grids."""
    support_fraction = 0.01
    n_common_bins = 400

    bands = tuple((float(a), float(b)) for a, b in diagnostics["bands"])
    theory_products = diagnostics["theory"]["products"]
    emu_kappa = np.asarray(diagnostics["emulation"]["kappa"], dtype=np.float64)
    all_l1 = np.asarray(diagnostics["emulation"]["all_l1"], dtype=np.float64)

    if all_l1.ndim != 3:
        raise ValueError("diagnostics['emulation']['all_l1'] must have shape (N, scale, kappa).")

    n_real, n_scales, _ = all_l1.shape
    if n_scales != len(bands):
        raise ValueError("Wavelet-band count is inconsistent with the emulation diagnostics.")

    emu_mean_l1 = np.nanmean(all_l1, axis=0)
    theory_moments = np.full((n_scales, 4), np.nan, dtype=np.float64)
    emu_moments = np.full((n_real, n_scales, 4), np.nan, dtype=np.float64)

    for scale_index, (theta1, theta2) in enumerate(bands):
        label = f"{theta1:g}-{theta2:g}"
        theory = theory_products[label]
        theory_kappa = np.asarray(theory["kappa"], dtype=np.float64)
        theory_l1 = np.asarray(theory["l1"], dtype=np.float64)

        theory_lo, theory_hi = _moment_informative_support(
            theory_kappa,
            theory_l1,
            fraction=support_fraction,
        )
        emu_lo, emu_hi = _moment_informative_support(
            emu_kappa[scale_index],
            emu_mean_l1[scale_index],
            fraction=support_fraction,
        )

        common_lo = max(theory_lo, emu_lo)
        common_hi = min(theory_hi, emu_hi)
        if common_lo >= common_hi:
            raise RuntimeError(f"No common informative support for {label} arcmin.")

        common_kappa = np.linspace(common_lo, common_hi, n_common_bins)
        if common_lo < 0.0 < common_hi:
            common_kappa = np.unique(
                np.sort(np.concatenate([common_kappa, np.asarray([0.0])]))
            )

        theory_common = _moment_interp_curve(
            theory_kappa,
            theory_l1,
            common_kappa,
        )
        zero = common_kappa == 0.0
        if np.any(zero):
            theory_common[zero] = 0.0
        theory_moments[scale_index] = l1_weighted_moments(
            common_kappa,
            theory_common,
        )

        for realisation in range(n_real):
            emu_common = _moment_interp_curve(
                emu_kappa[scale_index],
                all_l1[realisation, scale_index],
                common_kappa,
            )
            if np.any(zero):
                emu_common[zero] = 0.0
            emu_moments[realisation, scale_index] = l1_weighted_moments(
                common_kappa,
                emu_common,
            )

    metric_indices = (1, 2, 3)
    metric_names = ("Variance", "Skewness", "Excess kurtosis")
    x = np.arange(n_scales, dtype=np.float64)
    scale_labels = [f"{a:g}-{b:g}" for a, b in bands]

    emu_mean = np.nanmean(emu_moments, axis=0)
    emu_low = np.nanpercentile(emu_moments, 2.5, axis=0)
    emu_high = np.nanpercentile(emu_moments, 97.5, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    for ax, metric_index, title in zip(axes, metric_indices, metric_names):
        theory_values = theory_moments[:, metric_index]
        emu_values = emu_mean[:, metric_index]
        lower_error = emu_values - emu_low[:, metric_index]
        upper_error = emu_high[:, metric_index] - emu_values

        ax.plot(
            x,
            theory_values,
            color="C0",
            marker="o",
            linestyle="-",
        )
        ax.errorbar(
            x,
            emu_values,
            yerr=[lower_error, upper_error],
            color="C1",
            marker="o",
            linestyle="--",
            capsize=4,
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.set_xlabel("Wavelet scale [arcmin]")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="C0",
            marker="o",
            linestyle="-",
            label="Theory (1 realisation)",
        ),
        Line2D(
            [0],
            [0],
            color="C1",
            marker="o",
            linestyle="--",
            label="Emulation",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    return fig, axes
