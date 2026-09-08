"""Core algorithms for WELCOME theory-driven full-sky weak-lensing map generation.

The module uses theoretical cosmological inputs only. Reference data are not required
for target construction or map generation.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import camb
import healpy as hp
import numpy as np
from astropy import units as u
from camb import model
from camb.sources import GaussianSourceWindow
from joblib import Parallel, delayed
from numpy.polynomial import Chebyshev
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid, trapezoid, simpson
from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline
from scipy.optimize import brentq, root, least_squares, newton
from scipy.special import eval_legendre, j1, ndtr, sici, spherical_jn
from scipy.stats import chi2, gaussian_kde, norm


@dataclass
class WelcomeConfig:
    """Configuration for a complete WELCOME run."""
    h: float = 0.70
    omega_b: float = 0.046
    omega_cdm: float = 0.233
    sigma8: float = 0.82
    n_s: float = 0.97
    single_plane: bool = True
    source_redshift: float = 1.0334
    nz_file: str | Path | None = None
    nside: int = 512
    lmax: int = 2000
    n_realisations: int = 100
    n_iterations: int = 30
    n_jobs: int = 5
    output_dir: str | Path = "WELCOME_output"

    # Advanced parameters.
    sum_mnu_ev: float = 0.0
    a_s_times_1e9: float = 2.10
    w0: float = -1.0
    wa: float = 0.0
    halofit_version: str = "mead2020"
    theory_nside: int = 4096
    theory_ell_min: int = 2
    theory_ell_max: int = 3000
    n_power_spectrum_bins: int = 40
    theory_random_seed: int = 12345
    random_seed: int = 14
    harmonic_iter: int = 3
    overwrite: bool = False


ACTIVE_CONFIG = None


def _apply_config(config: WelcomeConfig) -> None:
    global ACTIVE_CONFIG
    global H, H0_NORMALIZATION, OMEGA_B, OMEGA_CDM, SUM_MNU_EV, N_S
    global A_S_TIMES_1E9, SIGMA8_TARGET, W_DE, WA_DE, HALOFIT_VERSION
    global NZ_FILE, SOURCE_PLANE_REDSHIFT, SINGLE_PLANE_CAMB_SIGMA_Z
    global LENS_SHELL_WIDTH_HMPC, THEORY_NSIDE, THEORY_ELL_MIN, THEORY_ELL_MAX
    global N_POWER_SPECTRUM_BINS, N_THEORY_POWER_SPECTRA, THEORY_RANDOM_SEED
    global F_SKY, NOISE_LEVEL, PSD_RELATIVE_FLOOR, CAMB_KMAX_MPC, PROFILE_KMAX_MPC
    global N_MEAN_Z, N_HALO_Z, N_MASS, LOG10_M_MIN, LOG10_M_MAX, SIGMA_K_MIN
    global SIGMA_K_MAX, SIGMA_NK, N_ANGLE, N_LONG, LONG_K_MIN, LONG_K_MAX
    global RESPONSE_DLOGK, CONC_SCATTER_LOG10, CONC_GH_ORDER, C_K_TIDAL
    global EMULATION_NSIDE, EMULATION_LMAX_TARGET, EMULATION_LMAX
    global REALISATION_START, REALISATION_END, N_JOBS, N_ITERATIONS, RANDOM_SEED
    global HARMONIC_ITER, OVERWRITE_EMULATION, EMULATION_L1_BANDS, EMULATION_COARSE_ARCMIN
    global K_MIN, K_MAX, N_K, COARSE_ARCMIN, HISTOGRAM_BINS
    global BASE_OUTPUT_DIR, L1_DIR, EMULATION_OUTPUT_DIR, TARGET_POWER_NPZ
    global THEORY_ALL_SCALE_NPZ, RUN_EMULATION, USE_PIXEL_WINDOW_IN_THEORY_VARIANCE
    global REAL_LAMBDA_POINTS, LAMBDA_SIGMA_FACTOR, LAMBDA_MIN_ABS_MAX, LAMBDA_MAX_ABS_MAX
    global CONTOUR_MAX, CONTOUR_STEP, POLYNOMIAL_DEGREE, THEORY_KAPPA_MIN_SIGMA
    global THEORY_KAPPA_MAX_SIGMA, THEORY_KAPPA_POINTS, SOLVER_RESIDUAL_TOL

    ACTIVE_CONFIG = config
    H=float(config.h); H0_NORMALIZATION=100.0; OMEGA_B=float(config.omega_b); OMEGA_CDM=float(config.omega_cdm)
    SUM_MNU_EV=float(config.sum_mnu_ev); N_S=float(config.n_s); A_S_TIMES_1E9=float(config.a_s_times_1e9)
    SIGMA8_TARGET=float(config.sigma8); W_DE=float(config.w0); WA_DE=float(config.wa); HALOFIT_VERSION=str(config.halofit_version)
    NZ_FILE=None if config.single_plane else (None if config.nz_file is None else Path(config.nz_file))
    SOURCE_PLANE_REDSHIFT=float(config.source_redshift); SINGLE_PLANE_CAMB_SIGMA_Z=0.005
    if not config.single_plane and NZ_FILE is None:
        raise ValueError("nz_file is required when single_plane=False.")
    LENS_SHELL_WIDTH_HMPC=150.0; THEORY_NSIDE=int(config.theory_nside); THEORY_ELL_MIN=int(config.theory_ell_min); THEORY_ELL_MAX=int(config.theory_ell_max)
    N_POWER_SPECTRUM_BINS=int(config.n_power_spectrum_bins); N_THEORY_POWER_SPECTRA=int(config.n_realisations); THEORY_RANDOM_SEED=int(config.theory_random_seed)
    F_SKY=1.0; NOISE_LEVEL=0.0; PSD_RELATIVE_FLOOR=1.0e-12
    CAMB_KMAX_MPC=100.0; PROFILE_KMAX_MPC=100.0; N_MEAN_Z=320; N_HALO_Z=80; N_MASS=192
    LOG10_M_MIN=8.0; LOG10_M_MAX=16.5; SIGMA_K_MIN=1.0e-5; SIGMA_K_MAX=100.0; SIGMA_NK=768
    N_ANGLE=24; N_LONG=700; LONG_K_MIN=1.0e-6; LONG_K_MAX=5.0e-2; RESPONSE_DLOGK=2.0e-3
    CONC_SCATTER_LOG10=0.15; CONC_GH_ORDER=3; C_K_TIDAL=1.0
    EMULATION_NSIDE=int(config.nside); EMULATION_LMAX_TARGET=int(config.lmax); EMULATION_LMAX=min(EMULATION_LMAX_TARGET,3*EMULATION_NSIDE-1)
    REALISATION_START=0; REALISATION_END=int(config.n_realisations)-1; N_JOBS=int(config.n_jobs); N_ITERATIONS=int(config.n_iterations)
    RANDOM_SEED=int(config.random_seed); HARMONIC_ITER=int(config.harmonic_iter); OVERWRITE_EMULATION=bool(config.overwrite)
    EMULATION_L1_BANDS=[(10.0,20.0),(20.0,40.0)]; EMULATION_COARSE_ARCMIN=40.0
    K_MIN=1.0e-4; K_MAX=50.0; N_K=512; COARSE_ARCMIN=40.0; HISTOGRAM_BINS=400
    USE_PIXEL_WINDOW_IN_THEORY_VARIANCE=False
    REAL_LAMBDA_POINTS=41; LAMBDA_SIGMA_FACTOR=1.20; LAMBDA_MIN_ABS_MAX=80.0; LAMBDA_MAX_ABS_MAX=600.0
    CONTOUR_MAX=40000.0; CONTOUR_STEP=10.0; POLYNOMIAL_DEGREE=9; THEORY_KAPPA_MIN_SIGMA=-10.0
    THEORY_KAPPA_MAX_SIGMA=10.0; THEORY_KAPPA_POINTS=300; SOLVER_RESIDUAL_TOL=2.0e-4
    BASE_OUTPUT_DIR=Path(config.output_dir); L1_DIR=BASE_OUTPUT_DIR/"wavelet_l1"
    EMULATION_OUTPUT_DIR=BASE_OUTPUT_DIR/"emulated_map"/f"nside{EMULATION_NSIDE}_lmax{EMULATION_LMAX}"
    TARGET_POWER_NPZ=BASE_OUTPUT_DIR/"theoretical_power_spectra.npz"
    for path in (BASE_OUTPUT_DIR,L1_DIR,EMULATION_OUTPUT_DIR): path.mkdir(parents=True,exist_ok=True)
    RUN_EMULATION=True



def load_nz_file(path):
    """Load and normalise a source redshift distribution from .npy."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'n(z) file not found:\n{path}')
    data = np.load(path, allow_pickle=False)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError('n(z) input must be a 2-D array containing z and n(z).')
    if data.shape[1] == 2:
        z = data[:, 0]
        nz = data[:, 1]
    elif data.shape[0] == 2:
        z = data[0]
        nz = data[1]
    else:
        raise ValueError('n(z) input must have shape (N, 2) or (2, N).')
    order = np.argsort(z)
    z = np.asarray(z[order], dtype=np.float64)
    nz = np.asarray(nz[order], dtype=np.float64)
    valid = np.isfinite(z) & np.isfinite(nz) & (z >= 0.0)
    z = z[valid]
    nz = np.clip(nz[valid], 0.0, None)
    if z.size < 2 or np.any(np.diff(z) <= 0.0):
        raise ValueError('The n(z) redshift grid must be strictly increasing.')
    return (z, nz)


def source_summary_from_distribution(z, nz):
    """Return useful summary redshifts without replacing the full n(z)."""
    mean_z = float(np.trapezoid(z * nz, z))
    variance_z = float(np.trapezoid((z - mean_z) ** 2 * nz, z))
    std_z = float(np.sqrt(max(variance_z, 0.0)))
    cumulative = cumulative_trapezoid(nz, z, initial=0.0)
    cumulative = cumulative / cumulative[-1]
    median_z = float(np.interp(0.5, cumulative, z))
    peak_z = float(z[np.argmax(nz)])
    return (mean_z, std_z, median_z, peak_z)


def configure_source_population(nz_file, source_plane_redshift):
    """Build the common source configuration used by the full notebook."""
    if nz_file is not None:
        z, nz = load_nz_file(nz_file)
        mean_z, std_z, median_z, peak_z = source_summary_from_distribution(z, nz)
        return {'mode': 'redshift_distribution', 'z': z, 'nz': nz, 'effective_redshift': mean_z, 'redshift_std': std_z, 'median_redshift': median_z, 'peak_redshift': peak_z, 'tag': Path(nz_file).stem, 'path': Path(nz_file)}
    z0 = float(source_plane_redshift)
    if not np.isfinite(z0) or z0 <= 0.0:
        raise ValueError('SOURCE_PLANE_REDSHIFT must be positive.')
    return {'mode': 'single_source_plane', 'z': None, 'nz': None, 'effective_redshift': z0, 'redshift_std': 0.0, 'median_redshift': z0, 'peak_redshift': z0, 'tag': f'source_plane_z{z0:.5f}'.replace('.', 'p'), 'path': None}




def normalize_pdf(kappa, pdf):
    kappa = np.asarray(kappa, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    pdf = np.clip(pdf, 0.0, None)
    norm = trapezoid(pdf, kappa)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f'Invalid PDF normalization: {norm}')
    return pdf / norm


def l1_from_pdf(pdf, kappa):
    return np.asarray(pdf, dtype=float) * np.abs(np.asarray(kappa, dtype=float))


def pdf_moments(kappa, pdf):
    kappa = np.asarray(kappa, dtype=float)
    pdf = normalize_pdf(kappa, pdf)
    mean = trapezoid(kappa * pdf, kappa)
    variance = trapezoid((kappa - mean) ** 2 * pdf, kappa)
    third = trapezoid((kappa - mean) ** 3 * pdf, kappa)
    fourth = trapezoid((kappa - mean) ** 4 * pdf, kappa)
    s3 = third / variance ** 2 if variance > 0 else np.nan
    excess_kurtosis = fourth / variance ** 2 - 3.0 if variance > 0 else np.nan
    return (mean, variance, s3, excess_kurtosis, trapezoid(pdf, kappa))


def tau_of_rho(rho):
    rho = np.asarray(rho, dtype=float)
    nu = 1.4
    return nu * (1.0 - rho ** (-1.0 / nu))


def theory_kappa_grid(theory_variance):
    sigma = np.sqrt(float(theory_variance))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f'Invalid theoretical sigma: {sigma}')
    return np.linspace(THEORY_KAPPA_MIN_SIGMA * sigma, THEORY_KAPPA_MAX_SIGMA * sigma, THEORY_KAPPA_POINTS)


class ThreeScaleCosmology:

    def __init__(self, h, H0, Ob, Oc, mnu, ns, As_1e9, w, wa, source_mode, source_plane_redshift=None, source_z=None, source_nz=None):
        self.h = float(h)
        self.H0 = float(H0)
        self.Ob = float(Ob)
        self.Oc = float(Oc)
        self.mnu = float(mnu)
        self.ns = float(ns)
        self.As = float(As_1e9) * 1e-09
        self.w = float(w)
        self.wa = float(wa)
        self.speed_light = 299792.458
        self.Omnu = self.mnu / 93.14 / self.h ** 2
        self.Om = self.Ob + self.Oc + self.Omnu
        self.Ol = 1.0 - self.Om
        self.source_mode = str(source_mode)
        self.source_plane_redshift = None if source_plane_redshift is None else float(source_plane_redshift)
        self.source_z = None if source_z is None else np.asarray(source_z, dtype=np.float64)
        self.source_nz = None if source_nz is None else np.asarray(source_nz, dtype=np.float64)
        if self.source_mode == 'redshift_distribution':
            if self.source_z is None or self.source_nz is None:
                raise ValueError('source_z and source_nz are required for n(z) mode.')
            self.zmax = max(5.0, float(self.source_z[-1]))
        elif self.source_mode == 'single_source_plane':
            if self.source_plane_redshift is None:
                raise ValueError('A source-plane redshift is required.')
            self.zmax = max(5.0, self.source_plane_redshift)
        else:
            raise ValueError(f'Unknown source mode: {self.source_mode}')
        self.pars, self.results = self._build_camb_with_target_sigma8()
        if self.source_mode == 'single_source_plane':
            self.source_chi_hmpc = float(self.get_chi(self.source_plane_redshift))
        else:
            positive = self.source_nz > 0.0
            zmax_source = float(self.source_z[positive][-1])
            self.source_chi_hmpc = float(self.get_chi(zmax_source))
        print('[COSMO] source mode=', self.source_mode, ', maximum source chi=', f'{self.source_chi_hmpc:.6f} Mpc/h')

    def _make_camb_params(self, As):
        pars = camb.set_params(H0=self.H0 * self.h, omch2=self.Oc * self.h ** 2, ombh2=self.Ob * self.h ** 2, mnu=self.mnu, omnuh2=self.Omnu * self.h ** 2, As=float(As), ns=self.ns, NonLinear=camb.model.NonLinear_both, halofit_version=HALOFIT_VERSION, w=self.w, wa=self.wa, dark_energy_model='fluid')
        pars.set_matter_power(redshifts=[0.0], kmax=20.0, nonlinear=False, silent=True)
        pars.WantTransfer = True
        return pars

    def _build_camb_with_target_sigma8(self):
        pars0 = self._make_camb_params(self.As)
        results0 = camb.get_results(pars0)
        sigma8_0 = float(np.asarray(results0.get_sigma8()).reshape(-1)[-1])
        if not np.isfinite(sigma8_0) or sigma8_0 <= 0.0:
            raise RuntimeError(f'CAMB returned invalid sigma8={sigma8_0}')
        As_matched = self.As * (float(SIGMA8_TARGET) / sigma8_0) ** 2
        pars = self._make_camb_params(As_matched)
        results = camb.get_results(pars)
        sigma8_final = float(np.asarray(results.get_sigma8()).reshape(-1)[-1])
        self.As = float(As_matched)
        self.sigma8_camb = sigma8_final
        print(f'[COSMO] CAMB amplitude match: sigma8(initial)={sigma8_0:.8f}, As={self.As:.8e}, sigma8(final)={sigma8_final:.8f}')
        return (pars, results)

    def get_chi(self, redshift):
        return self.results.comoving_radial_distance(redshift, tol=1e-07) * self.h

    def get_z_from_chi(self, chi):
        return self.results.redshift_at_comoving_radial_distance(np.asarray(chi, dtype=float) / self.h)

    def get_lensing_weight_array_source_plane(self, chis, chi_source=None):
        chis = np.asarray(chis, dtype=float)
        if chi_source is None:
            chi_source = self.source_chi_hmpc
        chi_source = float(chi_source)
        z_lens = np.asarray(self.get_z_from_chi(chis), dtype=float)
        prefactor = 1.5 * self.Om * (self.H0 / self.speed_light) ** 2
        geometry = np.zeros_like(chis)
        valid = (chis > 0.0) & (chis < chi_source)
        geometry[valid] = (chi_source - chis[valid]) / chi_source
        weights = prefactor * chis * (1.0 + z_lens) * geometry
        return (z_lens, weights)

    def get_lensing_weight_array_nz(self, chis):
        """Lensing kernel for an arbitrary normalised source n(z)."""
        chis = np.asarray(chis, dtype=np.float64)
        z_lens = np.asarray(self.get_z_from_chi(chis), dtype=np.float64)
        prefactor = 1.5 * self.Om * (self.H0 / self.speed_light) ** 2
        if self.source_mode == 'single_source_plane':
            return self.get_lensing_weight_array_source_plane(chis)
        z_source = self.source_z
        nz_source = self.source_nz
        chi_source = np.asarray(self.get_chi(z_source), dtype=np.float64)
        geometry = np.zeros_like(chis)
        for i, chi_lens in enumerate(chis):
            valid = chi_source > chi_lens
            if np.count_nonzero(valid) < 2:
                continue
            kernel = nz_source[valid] * (chi_source[valid] - chi_lens) / chi_source[valid]
            geometry[i] = np.trapezoid(kernel, z_source[valid])
        weights = prefactor * chis * (1.0 + z_lens) * geometry
        return (z_lens, weights)

    def get_matter_power_interpolator(self, nonlinear, kmin, kmax, nk):
        self.k_values = np.logspace(np.log10(kmin), np.log10(kmax), int(nk))
        return camb.get_matter_power_interpolator(self.pars, nonlinear=bool(nonlinear), hubble_units=True, k_hunit=True, kmax=float(kmax), zmax=self.zmax)


class ThreeScaleVariance:
    """
    Covariances used inside the LDT rate function.

    The contraction/rate function uses the LINEAR spectrum. Nonlinear small-scale
    power is introduced afterwards through a model-based variance rescaling,
    avoiding a fully nonlinear P(k) inside the initial Gaussian rate function.
    """

    def __init__(self, cosmo, pk_linear, pk_nonlinear):
        self.cosmo = cosmo
        self.pk_linear = pk_linear
        self.pk_nonlinear = pk_nonlinear

    @staticmethod
    def top_hat_window(x):
        x = np.asarray(x, dtype=float)
        result = np.ones_like(x)
        nonzero = np.abs(x) > 1e-12
        result[nonzero] = 2.0 * j1(x[nonzero]) / x[nonzero]
        return result

    def sigma2(self, redshift, R1, R2=None, *, nonlinear=False):
        if R2 is None:
            R2 = R1
        k = self.cosmo.k_values
        pk_object = self.pk_nonlinear if nonlinear else self.pk_linear
        pk = pk_object.P(float(redshift), k)
        window = self.top_hat_window(k * float(R1)) * self.top_hat_window(k * float(R2))
        return float(simpson(k * pk * window / (2.0 * np.pi), x=k))

    def linear_sigma2(self, redshift, R1, R2=None):
        return self.sigma2(redshift, R1, R2, nonlinear=False)

    def nonlinear_sigma2(self, redshift, R1, R2=None):
        return self.sigma2(redshift, R1, R2, nonlinear=True)

    def wavelet_slice_variance(self, redshift, R1, R2, *, nonlinear=False):
        return self.sigma2(redshift, R1, nonlinear=nonlinear) + self.sigma2(redshift, R2, nonlinear=nonlinear) - 2.0 * self.sigma2(redshift, R1, R2, nonlinear=nonlinear)


def theory_top_hat_beam(theta_arcmin, lmax):
    """
    Spherical top-hat definition used consistently for map smoothing and theory calculations.
    This filter is defined by the configured angular scale and is independent of any external data.
    """
    theta_radian = np.deg2rad(float(theta_arcmin) / 60.0)
    beta = np.linspace(0.0, 1.2 * theta_radian, 12000)
    normalization = 2.0 * np.pi * (1.0 - np.cos(theta_radian))
    profile = np.where(beta <= theta_radian, 1.0 / normalization, 0.0)
    return hp.beam2bl(profile, beta, lmax=lmax)


@dataclass
class ThreeScaleTheoryContext:
    cosmo: object
    variance: object
    chis: np.ndarray
    dchis: np.ndarray
    z_lens: np.ndarray
    lensing_weight: np.ndarray

    @classmethod
    def build(cls):
        cosmo = ThreeScaleCosmology(h=H, H0=H0_NORMALIZATION, Ob=OMEGA_B, Oc=OMEGA_CDM, mnu=SUM_MNU_EV, ns=N_S, As_1e9=A_S_TIMES_1E9, w=W_DE, wa=WA_DE, source_mode=SOURCE_MODE, source_plane_redshift=SOURCE_PLANE_REDSHIFT if SOURCE_MODE == 'single_source_plane' else None, source_z=SOURCE_Z_GRID, source_nz=SOURCE_NZ)
        chi_max = float(cosmo.source_chi_hmpc)
        shell_width = float(LENS_SHELL_WIDTH_HMPC)
        edges = np.arange(0.0, chi_max, shell_width, dtype=np.float64)
        if edges.size == 0 or edges[0] != 0.0:
            edges = np.r_[0.0, edges]
        if edges[-1] < chi_max:
            edges = np.r_[edges, chi_max]
        dchis = np.diff(edges)
        chis = 0.5 * (edges[:-1] + edges[1:])
        z_lens, lensing_weight = cosmo.get_lensing_weight_array_nz(chis)
        print('[LOS] shell centres [Mpc/h]:', chis)
        print('[LOS] shell widths [Mpc/h]:', dchis)
        print('[LOS] lens redshifts:', z_lens)
        pk_linear = cosmo.get_matter_power_interpolator(False, K_MIN, K_MAX, N_K)
        pk_nonlinear = cosmo.get_matter_power_interpolator(True, K_MIN, K_MAX, N_K)
        variance = ThreeScaleVariance(cosmo, pk_linear, pk_nonlinear)
        context = cls(cosmo, variance, chis, dchis, z_lens, lensing_weight)
        context._angular_cl_cache = {}
        return context

    def wavelet_linear_variance(self, theta1_arcmin, theta2_arcmin):
        theta1 = (theta1_arcmin * u.arcmin).to(u.radian).value
        theta2 = (theta2_arcmin * u.arcmin).to(u.radian).value
        values = np.asarray([self.variance.wavelet_slice_variance(z, chi * theta1, chi * theta2, nonlinear=False) for z, chi in zip(self.z_lens, self.chis)], dtype=float)
        return float(np.sum(self.dchis * self.lensing_weight ** 2 * values))

    def coarse_linear_variance(self, theta_arcmin):
        theta = (theta_arcmin * u.arcmin).to(u.radian).value
        values = np.asarray([self.variance.linear_sigma2(z, chi * theta) for z, chi in zip(self.z_lens, self.chis)], dtype=float)
        return float(np.sum(self.dchis * self.lensing_weight ** 2 * values))

    def angular_cl(self, *, nonlinear=True, lmax=None):
        if lmax is None:
            lmax = 3 * int(THEORY_NSIDE) - 1
        key = (bool(nonlinear), int(lmax))
        if key in self._angular_cl_cache:
            return self._angular_cl_cache[key].copy()
        ells = np.arange(lmax + 1, dtype=float)
        cl = np.zeros(lmax + 1, dtype=float)
        pk_object = self.variance.pk_nonlinear if nonlinear else self.variance.pk_linear
        for chi, dchi, z, weight in zip(self.chis, self.dchis, self.z_lens, self.lensing_weight):
            if chi <= 0.0 or weight == 0.0:
                continue
            k = (ells + 0.5) / chi
            valid = (k >= K_MIN) & (k <= K_MAX)
            plane_power = np.zeros_like(k)
            if np.any(valid):
                plane_power[valid] = pk_object.P(float(z), k[valid])
            cl += dchi * weight ** 2 * plane_power / chi ** 2
        self._angular_cl_cache[key] = cl.copy()
        return cl

    def _pixel_window(self, lmax):
        if not USE_PIXEL_WINDOW_IN_THEORY_VARIANCE:
            return np.ones(lmax + 1, dtype=float)
        return np.asarray(hp.pixwin(THEORY_NSIDE, lmax=lmax), dtype=float)

    def wavelet_variance(self, theta1_arcmin, theta2_arcmin):
        lmax = 3 * int(THEORY_NSIDE) - 1
        ells = np.arange(lmax + 1, dtype=float)
        cl = self.angular_cl(nonlinear=True, lmax=lmax)
        pixel = self._pixel_window(lmax)
        beam1 = theory_top_hat_beam(theta1_arcmin, lmax)
        beam2 = theory_top_hat_beam(theta2_arcmin, lmax)
        total_filter = pixel * (beam1 - beam2)
        return float(np.sum((2.0 * ells + 1.0) / (4.0 * np.pi) * cl * total_filter ** 2))

    def coarse_variance(self, theta_arcmin):
        lmax = 3 * int(THEORY_NSIDE) - 1
        ells = np.arange(lmax + 1, dtype=float)
        cl = self.angular_cl(nonlinear=True, lmax=lmax)
        pixel = self._pixel_window(lmax)
        beam = theory_top_hat_beam(theta_arcmin, lmax)
        total_filter = pixel * beam
        return float(np.sum((2.0 * ells + 1.0) / (4.0 * np.pi) * cl * total_filter ** 2))


def psi_2cell_scalar(context, chi, z, delta1, delta2, theta1_rad, theta2_rad):
    delta1 = float(delta1)
    delta2 = float(delta2)
    if delta1 <= -0.9999 or delta2 <= -0.9999:
        return 1e+100
    rho1 = 1.0 + delta1
    rho2 = 1.0 + delta2
    tau1 = float(tau_of_rho(rho1))
    tau2 = float(tau_of_rho(rho2))
    radius1 = float(chi) * np.sqrt(rho1) * float(theta1_rad)
    radius2 = float(chi) * np.sqrt(rho2) * float(theta2_rad)
    c11 = context.variance.linear_sigma2(z, radius1, radius1)
    c12 = context.variance.linear_sigma2(z, radius1, radius2)
    c22 = context.variance.linear_sigma2(z, radius2, radius2)
    determinant = c11 * c22 - c12 ** 2
    if not np.isfinite(determinant) or determinant <= 0.0:
        return 1e+100
    numerator = c11 * tau2 ** 2 - 2.0 * c12 * tau1 * tau2 + c22 * tau1 ** 2
    return numerator / (2.0 * determinant)


def solve_2cell_delta(context, lambda_value, lens_weight, chi, z, theta1_rad, theta2_rad, initial_guess):
    """
    Robust stationary-point solver for the two-cell rate function.

    The original implementation used a one-sided derivative with one fixed
    delta-step.  The smallest angular scale and the very first lens plane are
    much more numerically stiff, so here we use central differences and retry
    several derivative steps before declaring a failure.
    """
    if np.isclose(lambda_value, 0.0, atol=1e-14):
        return np.asarray([0.0, 0.0], dtype=float)
    derivative_steps = (1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005)
    forcing_scale = abs(float(lambda_value) * float(lens_weight))
    residual_tolerance = 0.0002
    guess0 = np.clip(np.asarray(initial_guess, dtype=float), -0.95, 10.0)
    fallback_guesses = [guess0, np.zeros(2, dtype=float), np.asarray([-0.05, 0.05]), np.asarray([-0.15, 0.15]), np.asarray([-0.3, 0.3]), np.asarray([0.15, -0.15])]
    best_x = None
    best_residual = np.inf
    best_step = None
    for derivative_step in derivative_steps:

        def equations(delta):
            d1 = float(delta[0])
            d2 = float(delta[1])
            h1 = min(derivative_step, 0.45 * (d1 + 0.999))
            h2 = min(derivative_step, 0.45 * (d2 + 0.999))
            h1 = max(h1, 1e-07)
            h2 = max(h2, 1e-07)
            psi_d1_plus = psi_2cell_scalar(context, chi, z, d1 + h1, d2, theta1_rad, theta2_rad)
            psi_d1_minus = psi_2cell_scalar(context, chi, z, d1 - h1, d2, theta1_rad, theta2_rad)
            psi_d2_plus = psi_2cell_scalar(context, chi, z, d1, d2 + h2, theta1_rad, theta2_rad)
            psi_d2_minus = psi_2cell_scalar(context, chi, z, d1, d2 - h2, theta1_rad, theta2_rad)
            derivative1 = (psi_d1_plus - psi_d1_minus) / (2.0 * h1)
            derivative2 = (psi_d2_plus - psi_d2_minus) / (2.0 * h2)
            result = np.asarray([float(lambda_value) * float(lens_weight) + derivative1, -float(lambda_value) * float(lens_weight) + derivative2], dtype=float)
            return np.nan_to_num(result, nan=1e+100, posinf=1e+100, neginf=-1e+100)
        hybrid = root(equations, guess0, method='hybr')
        if hybrid.success and np.all(np.isfinite(hybrid.x)) and np.all(hybrid.x > -0.999):
            residual = float(np.linalg.norm(equations(hybrid.x)))
            if residual < best_residual:
                best_x = hybrid.x.copy()
                best_residual = residual
                best_step = derivative_step
            if residual < residual_tolerance:
                return hybrid.x.astype(float)
        for fallback_guess in fallback_guesses:
            solution = least_squares(equations, x0=np.clip(fallback_guess, -0.95, 10.0), bounds=([-0.999, -0.999], [30.0, 30.0]), x_scale='jac', loss='linear', xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=1200)
            residual = float(np.linalg.norm(equations(solution.x)))
            if residual < best_residual:
                best_x = solution.x.copy()
                best_residual = residual
                best_step = derivative_step
            if solution.success and residual < residual_tolerance:
                return solution.x.astype(float)
        if best_x is not None:
            guess0 = np.clip(best_x, -0.95, 10.0)
            fallback_guesses[0] = guess0
    raise RuntimeError(f'2-cell solve failed after adaptive central differences: lambda={lambda_value}, z={z}, chi={chi}, lens_weight={lens_weight}, best_delta={best_x}, best_residual={best_residual}, accepted_tolerance={residual_tolerance}, forcing_scale={forcing_scale}, best_derivative_step={best_step}')


def projected_cgf_2cell(context, theta1_arcmin, theta2_arcmin, lambdas):
    lambdas = np.asarray(lambdas, dtype=float)
    theta1_rad = (theta1_arcmin * u.arcmin).to(u.radian).value
    theta2_rad = (theta2_arcmin * u.arcmin).to(u.radian).value
    nplanes = len(context.chis)
    zero_index = int(np.argmin(np.abs(lambdas)))
    deltas = np.zeros((nplanes, len(lambdas), 2), dtype=float)
    maximum_weight = np.max(np.abs(context.lensing_weight))
    negligible_weight = maximum_weight * 1e-07
    for plane in range(nplanes):
        print(f'[THEORY 2-cell] plane {plane + 1}/{nplanes}', end='\r')
        if np.abs(context.lensing_weight[plane]) <= negligible_weight:
            continue
        for index in range(zero_index + 1, len(lambdas)):
            deltas[plane, index] = solve_2cell_delta(context, lambdas[index], context.lensing_weight[plane], context.chis[plane], context.z_lens[plane], theta1_rad, theta2_rad, deltas[plane, index - 1])
        for index in range(zero_index - 1, -1, -1):
            deltas[plane, index] = solve_2cell_delta(context, lambdas[index], context.lensing_weight[plane], context.chis[plane], context.z_lens[plane], theta1_rad, theta2_rad, deltas[plane, index + 1])
    print()
    phi = np.zeros(len(lambdas), dtype=float)
    for index, lambda_value in enumerate(lambdas):
        for plane in range(nplanes):
            delta1, delta2 = deltas[plane, index]
            psi = psi_2cell_scalar(context, context.chis[plane], context.z_lens[plane], delta1, delta2, theta1_rad, theta2_rad)
            phi[index] += (lambda_value * context.lensing_weight[plane] * (delta1 - delta2) - psi) * context.dchis[plane]
    phi -= phi[zero_index]
    return phi


def psi_1cell_scalar(context, chi, z, delta, theta_rad):
    delta = float(delta)
    if delta <= -0.9999:
        return 1e+100
    rho = 1.0 + delta
    radius = float(chi) * np.sqrt(rho) * float(theta_rad)
    sigma2 = context.variance.linear_sigma2(z, radius)
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return 1e+100
    return float(tau_of_rho(rho) ** 2) / (2.0 * sigma2)


def solve_1cell_delta(context, lambda_value, lens_weight, chi, z, theta_rad, initial_guess):
    if np.isclose(lambda_value, 0.0, atol=1e-14):
        return 0.0
    derivative_step = 1e-05

    def equation(delta_array):
        delta = float(delta_array[0])
        psi_plus = psi_1cell_scalar(context, chi, z, delta + derivative_step, theta_rad)
        psi_minus = psi_1cell_scalar(context, chi, z, max(delta - derivative_step, -0.9998), theta_rad)
        derivative = (psi_plus - psi_minus) / (2.0 * derivative_step)
        return np.array([float(lambda_value) * float(lens_weight) - derivative], dtype=float)
    guesses = [float(initial_guess), 0.0, -0.2, 0.2, -0.5, 0.5]
    best_x = None
    best_residual = np.inf
    for guess in guesses:
        solution = least_squares(equation, x0=[np.clip(guess, -0.95, 10.0)], bounds=(-0.999, 30.0), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=400)
        residual = abs(float(equation(solution.x)[0]))
        if residual < best_residual:
            best_x = float(solution.x[0])
            best_residual = float(residual)
        if solution.success and residual < SOLVER_RESIDUAL_TOL:
            return float(solution.x[0])
    if best_x is not None and np.isfinite(best_x) and np.isfinite(best_residual) and (best_residual < SOLVER_RESIDUAL_TOL):
        return float(best_x)
    raise RuntimeError(f'1-cell solve failed: lambda={lambda_value}, z={z}, best_delta={best_x}, residual={best_residual}, tolerance={SOLVER_RESIDUAL_TOL}')


def projected_cgf_1cell(context, theta_arcmin, lambdas):
    lambdas = np.asarray(lambdas, dtype=float)
    theta_rad = (theta_arcmin * u.arcmin).to(u.radian).value
    nplanes = len(context.chis)
    zero_index = int(np.argmin(np.abs(lambdas)))
    deltas = np.zeros((nplanes, len(lambdas)), dtype=float)
    maximum_weight = np.max(np.abs(context.lensing_weight))
    negligible_weight = maximum_weight * 1e-07
    for plane in range(nplanes):
        print(f'[THEORY 1-cell] plane {plane + 1}/{nplanes}', end='\r')
        if np.abs(context.lensing_weight[plane]) <= negligible_weight:
            continue
        for index in range(zero_index + 1, len(lambdas)):
            deltas[plane, index] = solve_1cell_delta(context, lambdas[index], context.lensing_weight[plane], context.chis[plane], context.z_lens[plane], theta_rad, deltas[plane, index - 1])
        for index in range(zero_index - 1, -1, -1):
            deltas[plane, index] = solve_1cell_delta(context, lambdas[index], context.lensing_weight[plane], context.chis[plane], context.z_lens[plane], theta_rad, deltas[plane, index + 1])
    print()
    phi = np.zeros(len(lambdas), dtype=float)
    for index, lambda_value in enumerate(lambdas):
        for plane in range(nplanes):
            delta = deltas[plane, index]
            psi = psi_1cell_scalar(context, context.chis[plane], context.z_lens[plane], delta, theta_rad)
            phi[index] += (lambda_value * context.lensing_weight[plane] * delta - psi) * context.dchis[plane]
    phi -= phi[zero_index]
    return phi


def scgf_to_pdf(kappa_grid, real_lambdas, scgf):
    real_lambdas = np.asarray(real_lambdas, dtype=float)
    scgf = np.asarray(scgf, dtype=float)
    kappa_grid = np.asarray(kappa_grid, dtype=float)
    derivative = CubicSpline(real_lambdas, scgf)(real_lambdas, 1)
    radicand = 2.0 * (real_lambdas * derivative - scgf)
    tolerance = max(np.nanmax(np.abs(radicand)) * 1e-10, 1e-14)
    valid = np.isfinite(radicand) & np.isfinite(derivative) & (radicand >= -tolerance)
    x_data = np.sign(real_lambdas[valid]) * np.sqrt(np.clip(radicand[valid], 0.0, None))
    y_data = derivative[valid]
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x_data, y_data = (x_data[finite], y_data[finite])
    order = np.argsort(x_data)
    x_data, y_data = (x_data[order], y_data[order])
    unique_x, unique_indices = np.unique(np.round(x_data, 14), return_index=True)
    x_data = x_data[unique_indices]
    y_data = y_data[unique_indices]
    degree = min(POLYNOMIAL_DEGREE, len(x_data) - 1)
    if degree < 3:
        raise RuntimeError('Too few valid SCGF points for analytic continuation.')
    polynomial = Chebyshev.fit(x_data, y_data, deg=degree, domain=[float(np.min(x_data)), float(np.max(x_data))])
    polynomial_derivative = polynomial.deriv()
    lambda_contour = 1j * np.arange(0.0, CONTOUR_MAX, CONTOUR_STEP)
    tau_contour = np.zeros_like(lambda_contour, dtype=np.complex128)
    for index in range(1, len(lambda_contour)):
        lambda_value = lambda_contour[index]

        def equation(tau):
            return tau - polynomial_derivative(tau) * lambda_value
        try:
            tau_contour[index] = newton(equation, x0=tau_contour[index - 1], tol=1e-10, maxiter=100)
        except (RuntimeError, OverflowError):
            tau_contour[index] = newton(equation, x0=tau_contour[index - 1], x1=tau_contour[index - 1] + (1e-06 + 1e-06j), tol=1e-10, maxiter=200)
    phi_contour = lambda_contour * polynomial(tau_contour) - 0.5 * tau_contour ** 2
    delta_lambda = abs(lambda_contour[1] - lambda_contour[0]) * 1j
    weights = np.full(len(lambda_contour), delta_lambda, dtype=np.complex128)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    exponent = -lambda_contour[:, None] * kappa_grid[None, :] + phi_contour[:, None]
    pdf = np.imag(np.sum(np.exp(exponent) * weights[:, None], axis=0) / np.pi)
    return normalize_pdf(kappa_grid, pdf)


def _rescaled_lambda_grid(target_variance, variance_ratio):
    """
    Build the FINAL lambda grid while guaranteeing that the RAW LDT solver
    never sees |lambda_raw| > LAMBDA_MAX_ABS_MAX.

    We use

        lambda_raw = variance_ratio * lambda_final

    and therefore enforce

        |lambda_final|
        <= LAMBDA_MAX_ABS_MAX / variance_ratio.

    This restores the original code's solver-safe |lambda| <= 600 domain.
    """
    sigma = np.sqrt(float(target_variance))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f'Invalid target sigma: {sigma}')
    ratio = max(float(variance_ratio), 1e-12)
    requested_final_max = np.clip(LAMBDA_SIGMA_FACTOR / sigma, LAMBDA_MIN_ABS_MAX, LAMBDA_MAX_ABS_MAX)
    raw_solver_max = float(LAMBDA_MAX_ABS_MAX)
    safe_final_max = min(requested_final_max, raw_solver_max / ratio)
    if not np.isfinite(safe_final_max) or safe_final_max <= 0.0:
        raise ValueError('Could not construct a safe lambda grid.')
    lambdas = np.linspace(-safe_final_max, safe_final_max, REAL_LAMBDA_POINTS)
    raw_lambdas = ratio * lambdas
    maximum_raw = float(np.max(np.abs(raw_lambdas)))
    if maximum_raw > raw_solver_max + 1e-10:
        raise RuntimeError(f'Internal lambda-grid safety check failed: max |lambda_raw|={maximum_raw} > {raw_solver_max}')
    print(f'[lambda-grid] requested final max={requested_final_max:.6f}, safe final max={safe_final_max:.6f}, max raw={maximum_raw:.6f}')
    return lambdas


def wavelet_theory_product(context, theta1_arcmin, theta2_arcmin, name):
    linear_variance = context.wavelet_linear_variance(theta1_arcmin, theta2_arcmin)
    target_variance = context.wavelet_variance(theta1_arcmin, theta2_arcmin)
    variance_ratio = target_variance / linear_variance
    lambdas = _rescaled_lambda_grid(target_variance, variance_ratio)
    raw_lambdas = variance_ratio * lambdas
    maximum_raw_lambda = float(np.max(np.abs(raw_lambdas)))
    if maximum_raw_lambda > LAMBDA_MAX_ABS_MAX + 1e-10:
        raise RuntimeError(f'Unsafe raw lambda reached wavelet theory product: {maximum_raw_lambda}')
    kappa = theory_kappa_grid(target_variance)
    print(f'[{name}] linear LDT variance={linear_variance:.8e}')
    print(f'[{name}] exact nonlinear observable variance={target_variance:.8e}')
    print(f'[{name}] variance-rescaling ratio={variance_ratio:.6f}')
    raw_scgf = projected_cgf_2cell(context, theta1_arcmin, theta2_arcmin, raw_lambdas)
    scgf = raw_scgf / variance_ratio
    pdf = scgf_to_pdf(kappa, lambdas, scgf)
    pdf_statistics = pdf_moments(kappa, pdf)
    return {'field_name': name, 'scales_arcmin': np.asarray([theta1_arcmin, theta2_arcmin], dtype=float), 'kappa': kappa, 'pdf': pdf, 'l1': l1_from_pdf(pdf, kappa), 'variance': float(pdf_statistics[1]), 'theory_variance': float(target_variance), 'linear_ldt_variance': float(linear_variance), 'variance_rescaling_ratio': float(variance_ratio), 'real_lambdas': lambdas, 'raw_lambdas': raw_lambdas, 'scgf': scgf, 'raw_scgf': raw_scgf, 'redshift': float(SOURCE_EFFECTIVE_REDSHIFT), 'sigma_z': float(SOURCE_REDSHIFT_STD), 'source_mode': str(SOURCE_MODE), 'corrections': np.asarray(['linear-rate-function', 'configured-source-population', 'configured-sigma8-camb-amplitude-match', 'theory-nonlinear-variance-rescaling', 'spherical-top-hat', 'healpix-pixel-window', 'configured-line-of-sight-shells', 'chebyshev-continuation'], dtype=str)}


def coarse_theory_product(context, theta_arcmin):
    linear_variance = context.coarse_linear_variance(theta_arcmin)
    target_variance = context.coarse_variance(theta_arcmin)
    variance_ratio = target_variance / linear_variance
    lambdas = _rescaled_lambda_grid(target_variance, variance_ratio)
    raw_lambdas = variance_ratio * lambdas
    maximum_raw_lambda = float(np.max(np.abs(raw_lambdas)))
    if maximum_raw_lambda > LAMBDA_MAX_ABS_MAX + 1e-10:
        raise RuntimeError(f'Unsafe raw lambda reached wavelet theory product: {maximum_raw_lambda}')
    kappa = theory_kappa_grid(target_variance)
    print(f'[coarse] linear LDT variance={linear_variance:.8e}')
    print(f'[coarse] exact nonlinear observable variance={target_variance:.8e}')
    print(f'[coarse] variance-rescaling ratio={variance_ratio:.6f}')
    raw_scgf = projected_cgf_1cell(context, theta_arcmin, raw_lambdas)
    scgf = raw_scgf / variance_ratio
    pdf = scgf_to_pdf(kappa, lambdas, scgf)
    pdf_statistics = pdf_moments(kappa, pdf)
    return {'field_name': 'coarse', 'scales_arcmin': np.asarray([theta_arcmin], dtype=float), 'kappa': kappa, 'pdf': pdf, 'l1': l1_from_pdf(pdf, kappa), 'variance': float(pdf_statistics[1]), 'theory_variance': float(target_variance), 'linear_ldt_variance': float(linear_variance), 'variance_rescaling_ratio': float(variance_ratio), 'real_lambdas': lambdas, 'raw_lambdas': raw_lambdas, 'scgf': scgf, 'raw_scgf': raw_scgf, 'redshift': float(SOURCE_EFFECTIVE_REDSHIFT), 'sigma_z': float(SOURCE_REDSHIFT_STD), 'source_mode': str(SOURCE_MODE), 'corrections': np.asarray(['linear-rate-function', 'configured-source-population', 'configured-sigma8-camb-amplitude-match', 'theory-nonlinear-variance-rescaling', 'spherical-top-hat', 'healpix-pixel-window', 'configured-line-of-sight-shells', 'chebyshev-continuation'], dtype=str)}


def top_hat_beam(theta_arcmin, lmax, n_beta=10000):
    theta = np.deg2rad(float(theta_arcmin) / 60.0)
    beta = np.linspace(0.0, 1.2 * theta, int(n_beta))
    norm = 2 * np.pi * (1 - np.cos(theta))
    profile = np.where(beta <= theta, 1.0 / norm, 0.0)
    return hp.beam2bl(profile, beta, lmax=lmax)




def scale_token(value):
    """Convert an angular scale to the established saved-file token."""
    return f'{float(value):.1f}'.replace('.', '_')


def source_token():
    """Return the saved-product source tag for either supported source mode."""
    if SOURCE_MODE == 'single_source_plane':
        return 'z' + f'{float(SOURCE_PLANE_REDSHIFT):.4f}'.replace('.', '_')
    if SOURCE_MODE == 'redshift_distribution':
        if NZ_FILE is None:
            raise ValueError("NZ_FILE is required when SOURCE_MODE='redshift_distribution'.")
        return Path(NZ_FILE).stem
    raise ValueError(f'Unknown SOURCE_MODE: {SOURCE_MODE}')


def theory_l1_path(theta1=None, theta2=None, coarse=False):
    """Path of a saved theoretical wavelet/coarse product."""
    tag = source_token()
    if coarse:
        return L1_DIR / f'coarse_theory_{tag}_theta{scale_token(COARSE_ARCMIN)}.npy'
    if theta1 is None or theta2 is None:
        raise ValueError('theta1 and theta2 are required for a wavelet product.')
    return L1_DIR / f'theory_{tag}_theta{scale_token(theta1)}_{scale_token(theta2)}.npy'


def save_dict(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data, allow_pickle=True)
    print('[SAVED]', path)


def load_dict(path):
    x = np.load(path, allow_pickle=True)
    return x.item() if isinstance(x, np.ndarray) and x.shape == () else x


def apply_axis_limits(ax, xlim=None, ylim=None):
    """
    Apply optional x/y limits to a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.

    xlim : tuple or None
        (xmin, xmax). None keeps automatic limits.

    ylim : tuple or None
        (ymin, ymax). None keeps automatic limits.
    """
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def trap_weights(x):
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def cov_to_corr(cov):
    cov = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    den = np.outer(d, d)
    out = np.zeros_like(cov)
    good = den > 0
    out[good] = cov[good] / den[good]
    np.fill_diagonal(out, 1.0)
    return out


def project_psd(matrix, relative_floor=None):
    if relative_floor is None:
        relative_floor = globals().get("PSD_RELATIVE_FLOOR", 1.0e-12)
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(evals))), np.finfo(float).tiny)
    evals = np.maximum(evals, relative_floor * scale)
    out = evecs @ np.diag(evals) @ evecs.T
    return 0.5 * (out + out.T)


def chisquare_copula_samples(mean, covariance, n_samples=None, seed=None):
    if n_samples is None:
        n_samples = int(globals().get("N_THEORY_POWER_SPECTRA", 100))
    if seed is None:
        seed = int(globals().get("THEORY_RANDOM_SEED", 12345))
    mean = np.asarray(mean, dtype=float)
    covariance = project_psd(covariance)
    var = np.diag(covariance)
    if np.any(mean <= 0.0) or np.any(var <= 0.0):
        raise RuntimeError('positive mean/variance required for scaled-chi-square sampling')
    df = 2.0 * mean ** 2 / var
    scale = var / (2.0 * mean)
    target_corr = cov_to_corr(covariance)
    latent_corr = project_psd(target_corr)
    d = np.sqrt(np.diag(latent_corr))
    latent_corr = latent_corr / np.outer(d, d)
    np.fill_diagonal(latent_corr, 1.0)
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(mean.size), latent_corr, size=n_samples, method='eigh')
    u = np.clip(norm.cdf(z), np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    samples = scale[None, :] * chi2.ppf(u, df=df[None, :])
    return (samples, df, scale, target_corr, latent_corr)


def top_hat_window(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    m = np.abs(x) > 0.001
    xr = x[m]
    out[m] = 3.0 * (np.sin(xr) - xr * np.cos(xr)) / xr ** 3
    xs = x[~m]
    out[~m] = 1.0 - xs ** 2 / 10.0 + xs ** 4 / 280.0
    return out


def j0_second(x):
    x = np.asarray(x, dtype=float)
    j0 = spherical_jn(0, x)
    j1 = spherical_jn(1, x)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-05
    out[small] = -1.0 / 3.0
    out[~small] = -j0[~small] + 2.0 * j1[~small] / x[~small]
    return out


def build_global_binning(ells):
    edges = np.rint(np.geomspace(ELL_MIN, ELL_MAX + 1, N_GLOBAL_BINS + 1)).astype(int)
    edges[0] = ELL_MIN
    edges[-1] = ELL_MAX + 1
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    B = np.zeros((N_GLOBAL_BINS, ells.size), dtype=float)
    eff = np.zeros(N_GLOBAL_BINS)
    lo = np.zeros(N_GLOBAL_BINS, dtype=int)
    hi = np.zeros(N_GLOBAL_BINS, dtype=int)
    modes = np.zeros(N_GLOBAL_BINS)
    for b in range(N_GLOBAL_BINS):
        mask = (ells >= edges[b]) & (ells < edges[b + 1])
        ll = ells[mask].astype(float)
        if ll.size == 0:
            raise RuntimeError(f'empty ell bin {b}')
        ww = 2.0 * ll + 1.0
        sw = ww.sum()
        B[b, mask] = ww / sw
        eff[b] = np.sum(ww * ll) / sw
        lo[b], hi[b] = (int(ll[0]), int(ll[-1]))
        modes[b] = sw
    return (edges, B, eff, lo, hi, modes)


def make_camb_params(As_1e9, kmax=None):
    if kmax is None:
        kmax = float(globals().get("CAMB_KMAX_MPC", 100.0))
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100.0 * H, ombh2=OMEGA_B * H ** 2, omch2=OMEGA_CDM * H ** 2, mnu=SUM_MNU_EV, omk=0.0, tau=0.054, neutrino_hierarchy='degenerate', num_massive_neutrinos=1 if SUM_MNU_EV > 0 else 0)
    pars.InitPower.set_params(As=As_1e9 * 1e-09, ns=N_S)
    pars.set_dark_energy(w=W_DE, wa=WA_DE, dark_energy_model='ppf')
    pars.set_matter_power(redshifts=np.linspace(0.0, max(3.0, SOURCE_Z), 100), kmax=kmax, nonlinear=True, silent=True)
    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version=HALOFIT_VERSION)
    pars.WantCls = False
    pars.WantTransfer = True
    return pars


def P_lin(z, k):
    k = np.asarray(k, dtype=float)
    flat = k.ravel()
    zz = np.full_like(flat, float(z))
    out = np.asarray(pk_lin.P(zz, flat, grid=False), dtype=float)
    return out.reshape(k.shape)


def P_nl(z, k):
    k = np.asarray(k, dtype=float)
    flat = k.ravel()
    zz = np.full_like(flat, float(z))
    out = np.asarray(pk_nl.P(zz, flat, grid=False), dtype=float)
    return out.reshape(k.shape)


def source_kernel(z):
    z = np.asarray(z, dtype=float)
    chi = np.asarray(results.comoving_radial_distance(z), dtype=float)
    chi_s = float(np.asarray(results.comoving_radial_distance(SOURCE_Z)).squeeze())
    W = 1.5 * OMEGA_M * (100.0 * H / C_KM_S) ** 2 * (1.0 + z) * chi * np.clip(1.0 - chi / chi_s, 0.0, None)
    return (chi, W, chi_s)


def sigma_mass(z):
    R = (3.0 * mass_phys / (4.0 * np.pi * rho_m0)) ** (1.0 / 3.0)
    P = P_lin(z, sigma_k)
    Win = top_hat_window(np.outer(sigma_k, R))
    integ = sigma_k[:, None] ** 3 * P[:, None] * Win ** 2 / (2.0 * np.pi ** 2)
    s2 = np.trapezoid(integ, x=np.log(sigma_k), axis=0)
    sig = np.sqrt(np.maximum(s2, 1e-300))
    dlns = np.gradient(np.log(sig), lnM, edge_order=2)
    return (sig, dlns)


def tinker08_dn_dlnM(z, sig, dlns):
    Delta = 200.0
    alpha = 10.0 ** (-(0.75 / np.log10(Delta / 75.0)) ** 1.2)
    A = 0.186 * (1.0 + z) ** (-0.14)
    a = 1.47 * (1.0 + z) ** (-0.06)
    b = 2.57 * (1.0 + z) ** (-alpha)
    c = 1.19
    fs = A * ((sig / b) ** (-a) + 1.0) * np.exp(-c / sig ** 2)
    return np.maximum(rho_m0 / mass_phys * fs * np.abs(dlns), 0.0)


def tinker10_bias(sig):
    nu = DELTA_C / sig
    ld = np.log10(200.0)
    xp = np.exp(-(4.0 / ld) ** 4)
    A = 1.0 + 0.24 * ld * xp
    C = 0.019 + 0.107 * ld + 0.19 * xp
    aa = 0.44 * ld - 0.88
    B = 0.183
    bb = 1.5
    cc = 2.4
    nupa = nu ** aa
    return 1.0 - A * nupa / (nupa + DELTA_C ** aa) + B * nu ** bb + C * nu ** cc


def cbar_duffy08(z):
    c = 10.14 * (mass_h / 2000000000000.0) ** (-0.081) * (1.0 + z) ** (-1.01)
    return np.clip(c, 1.0, 40.0)


def concentration_nodes(cbar):
    x, w = hermgauss(CONC_GH_ORDER)
    sigln = CONC_SCATTER_LOG10 * np.log(10.0)
    cn = cbar[None, :] * np.exp(np.sqrt(2.0) * sigln * x[:, None])
    cn = np.clip(cn, 0.5, 80.0)
    return (cn, w / np.sqrt(np.pi))


def nfw_u_comoving(k, c):
    k = np.asarray(k, dtype=float)
    c = np.asarray(c, dtype=float)
    R200 = (3.0 * mass_phys / (4.0 * np.pi * 200.0 * rho_m0)) ** (1.0 / 3.0)
    rs = R200 / c
    x = np.outer(k, rs)
    cx = x * c[None, :]
    sx, cxint = sici(x)
    s1, c1 = sici((1.0 + c[None, :]) * x)
    safe = np.where(np.abs(x) > 1e-12, x, 1.0)
    num = np.sin(x) * (s1 - sx) + np.cos(x) * (c1 - cxint) - np.sin(cx) / ((1.0 + c[None, :]) * safe)
    den = np.log1p(c) - c / (1.0 + c)
    u = num / den[None, :]
    u[np.abs(x) < 1e-05] = 1.0
    return u


def halo_state(z):
    sig, dlns = sigma_mass(z)
    dn = tinker08_dn_dlnM(z, sig, dlns)
    bias = tinker10_bias(sig)
    base = wlnM * dn
    eps_m = 1.0 - np.sum(base * qM)
    eps_b = 1.0 - np.sum(base * qM * bias)
    return (base, bias, eps_m, eps_b, concentration_nodes(cbar_duffy08(z)))


def halo_integrals(z, k, state=None, need_matrices=True, need_conc_derivs=True):
    """Matter-profile halo integrals with low-mass delta completion.

    Returns center quantities for arbitrary k array:
      I11(k), I02diag(k), I11c(k), I02cdiag(k)
    If need_matrices:
      I12(k_i,k_j), I13[row=j,col=i]=I^1_3(k_i,k_j,k_j), I04(k_i,k_j)
    """
    k = np.asarray(k, dtype=float)
    if state is None:
        state = halo_state(z)
    base, bias, eps_m, eps_b, (cnodes, cweights) = state
    nk = k.size
    I11 = np.zeros(nk)
    I02d = np.zeros(nk)
    I11c = np.zeros(nk)
    I02cd = np.zeros(nk)
    if need_matrices:
        I12 = np.zeros((nk, nk))
        I13 = np.zeros((nk, nk))
        I04 = np.zeros((nk, nk))
    dc = 0.001
    for cnode, wc in zip(cnodes, cweights):
        u = nfw_u_comoving(k, cnode)
        if need_conc_derivs:
            up = nfw_u_comoving(k, cnode * np.exp(dc))
            um = nfw_u_comoving(k, cnode * np.exp(-dc))
            du = (up - um) / (2.0 * dc)
        else:
            du = np.zeros_like(u)
        I11 += wc * np.sum(u * (base * qM * bias)[None, :], axis=1)
        I02d += wc * np.sum(u ** 2 * (base * qM ** 2)[None, :], axis=1)
        I11c += wc * np.sum(du * (base * qM * bias)[None, :], axis=1)
        I02cd += wc * 2.0 * np.sum(u * du * (base * qM ** 2)[None, :], axis=1)
        if need_matrices:
            wb12 = base * qM ** 2 * bias
            wb13 = base * qM ** 3 * bias
            wb04 = base * qM ** 4
            I12 += wc * (u * np.sqrt(np.maximum(wb12, 0.0))[None, :] @ (u * np.sqrt(np.maximum(wb12, 0.0))[None, :]).T)
            I13 += wc * np.einsum('im,jm,m->ji', u, u ** 2, wb13, optimize=True)
            I04 += wc * (u ** 2 * np.sqrt(np.maximum(wb04, 0.0))[None, :] @ (u ** 2 * np.sqrt(np.maximum(wb04, 0.0))[None, :]).T)
        u0 = u[:, 0]
        du0 = du[:, 0]
        I11 += wc * eps_b * u0
        I02d += wc * eps_m * qmin * u0 ** 2
        I11c += wc * eps_b * du0
        I02cd += wc * 2.0 * eps_m * qmin * u0 * du0
        if need_matrices:
            I12 += wc * eps_b * qmin * np.outer(u0, u0)
            I13 += wc * eps_b * qmin ** 2 * np.einsum('i,j->ji', u0, u0 ** 2)
            I04 += wc * eps_m * qmin ** 3 * np.outer(u0 ** 2, u0 ** 2)
    out = dict(I11=I11, I02=I02d, I11c=I11c, I02c=I02cd, eps_m=eps_m, eps_b=eps_b)
    if need_matrices:
        out.update(I12=I12, I13=I13, I04=I04)
    return out


def safe_Plin(z, k):
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(k)
    good = (k >= SIGMA_K_MIN) & (k <= PROFILE_KMAX_MPC)
    if np.any(good):
        out[good] = P_lin(z, k[good])
    return out


def f2_ccl_pair(krow, kcol, theta):
    """CCL/Takada-Hu transformed F2 used in isotropized parallelogram trispectrum."""
    mu = np.cos(theta)
    kk = kcol[None, :]
    kp = krow[:, None]
    kr2 = kk ** 2 + kp ** 2 + 2.0 * kk * kp * mu
    kr = np.sqrt(np.maximum(kr2, 0.0))
    tiny = kr2 < 1e-28
    den = np.where(tiny, 1.0, kr2)
    term = 1.0 + kp / kk * mu
    f2 = 5.0 / 7.0 - 0.5 * (1.0 + kk ** 2 / den) * term + 2.0 / 7.0 * kk ** 2 / den * term ** 2
    f2[tiny] = 13.0 / 28.0
    return (kr, f2)


def isotropic_pt_pieces(z, k):
    k = np.asarray(k, dtype=float)
    nk = k.size
    pk = safe_Plin(z, k)
    prow = pk[None, :]
    P_iso = np.zeros((nk, nk))
    P3 = np.zeros((nk, nk))
    P4A = np.zeros((nk, nk))
    P4X = np.zeros((nk, nk))
    Xint = np.zeros((nk, nk))
    kk = k[None, :]
    kp = k[:, None]
    r = kp / kk
    for th, wt in zip(theta_nodes, theta_weights):
        mu = np.cos(th)
        kr, f2 = f2_ccl_pair(k, k, th)
        pkr = safe_Plin(z, kr)
        P_iso += wt * pkr
        P3 += wt * pkr * f2
        P4A += wt * pkr * f2 ** 2
        P4X += wt * pkr * f2 * f2.T
        den = 1.0 + r ** 2 + 2.0 * r * mu
        den = np.where(np.abs(den) < 1e-14, np.inf, den)
        xterm = (5.0 * r + (7.0 - 2.0 * r ** 2) * mu) / den * (3.0 / 7.0 * r + 0.5 * (1.0 + r ** 2) * mu + 4.0 / 7.0 * r * mu ** 2)
        Xint += wt * xterm
    Bpt = 6.0 / 7.0 * prow * prow.T + 2.0 * prow * P3
    Bpt = Bpt + Bpt.T
    X = -7.0 / 4.0 * (1.0 + r ** 2) + Xint
    t1113 = 4.0 / 9.0 * prow ** 2 * prow.T * X
    t1113 = t1113 + t1113.T
    t1122 = 8.0 * (prow ** 2 * P4A + prow * prow.T * P4X)
    t1122 = t1122 + t1122.T
    Tpt = t1113 + t1122
    return (P_iso, Bpt, Tpt)


def ssc_from_F(F):
    C = np.einsum('p,ip,jp->ij', wp, F, F, optimize=True) / (4.0 * np.pi)
    return 0.5 * (C + C.T)


def stochastic_spectrum_metrics(cov, corr):
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    eig = np.clip(eig, 0.0, None)
    s = eig.sum()
    pc1 = eig[-1] / s if s > 0 else np.nan
    p = eig / s if s > 0 else eig
    eff_rank = 1.0 / np.sum(p[p > 0] ** 2) if np.any(p > 0) else np.nan
    off = ~np.eye(corr.shape[0], dtype=bool)
    return (pc1, eff_rank, np.sqrt(np.mean(corr[off] ** 2)), np.mean(np.abs(corr[off])), np.max(np.abs(corr[off])))


def load_npz_product(path):
    """Load an NPZ product into a dictionary of NumPy arrays."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def save_theory_power_product(path, **arrays):
    """Save the complete theoretical power-spectrum product."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
def solve_2cell_delta(context, lambda_value, lens_weight, chi, z, theta1_rad, theta2_rad, initial_guess):
    if np.isclose(lambda_value, 0.0, atol=1e-14):
        return np.asarray([0.0, 0.0], dtype=float)
    derivative_steps = (1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005)
    forcing_scale = abs(float(lambda_value) * float(lens_weight))
    residual_tolerance = 0.0002
    guess0 = np.clip(np.asarray(initial_guess, dtype=float), -0.95, 10.0)
    fallback_guesses = [guess0, np.zeros(2, dtype=float), np.asarray([-0.05, +0.05], dtype=float), np.asarray([-0.15, +0.15], dtype=float), np.asarray([-0.3, +0.3], dtype=float), np.asarray([+0.15, -0.15], dtype=float)]
    best_x = None
    best_residual = np.inf
    best_step = None
    for derivative_step in derivative_steps:

        def equations(delta):
            d1 = float(delta[0])
            d2 = float(delta[1])
            h1 = min(derivative_step, 0.45 * (d1 + 0.999))
            h2 = min(derivative_step, 0.45 * (d2 + 0.999))
            h1 = max(h1, 1e-07)
            h2 = max(h2, 1e-07)
            psi_d1_plus = psi_2cell_scalar(context, chi, z, d1 + h1, d2, theta1_rad, theta2_rad)
            psi_d1_minus = psi_2cell_scalar(context, chi, z, d1 - h1, d2, theta1_rad, theta2_rad)
            psi_d2_plus = psi_2cell_scalar(context, chi, z, d1, d2 + h2, theta1_rad, theta2_rad)
            psi_d2_minus = psi_2cell_scalar(context, chi, z, d1, d2 - h2, theta1_rad, theta2_rad)
            derivative1 = (psi_d1_plus - psi_d1_minus) / (2.0 * h1)
            derivative2 = (psi_d2_plus - psi_d2_minus) / (2.0 * h2)
            result = np.asarray([float(lambda_value) * float(lens_weight) - derivative1, -float(lambda_value) * float(lens_weight) - derivative2], dtype=float)
            return np.nan_to_num(result, nan=1e+100, posinf=1e+100, neginf=-1e+100)
        hybrid = root(equations, guess0, method='hybr')
        if hybrid.success and np.all(np.isfinite(hybrid.x)) and np.all(hybrid.x > -0.999):
            residual = float(np.linalg.norm(equations(hybrid.x)))
            if residual < best_residual:
                best_x = hybrid.x.copy()
                best_residual = residual
                best_step = derivative_step
            if residual < residual_tolerance:
                return hybrid.x.astype(float)
        for fallback_guess in fallback_guesses:
            solution = least_squares(equations, x0=np.clip(fallback_guess, -0.95, 10.0), bounds=([-0.999, -0.999], [30.0, 30.0]), x_scale='jac', loss='linear', xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=1200)
            residual = float(np.linalg.norm(equations(solution.x)))
            if residual < best_residual:
                best_x = solution.x.copy()
                best_residual = residual
                best_step = derivative_step
            if solution.success and residual < residual_tolerance:
                return solution.x.astype(float)
        if best_x is not None:
            guess0 = np.clip(best_x, -0.95, 10.0)
            fallback_guesses[0] = guess0
    raise RuntimeError(f'2-cell solve failed after adaptive central differences: lambda={lambda_value}, z={z}, chi={chi}, lens_weight={lens_weight}, best_delta={best_x}, best_residual={best_residual}, accepted_tolerance={residual_tolerance}, forcing_scale={forcing_scale}, best_derivative_step={best_step}')


def projected_cgf_2cell(context, theta1_arcmin, theta2_arcmin, lambdas):
    lambdas = np.asarray(lambdas, dtype=float)
    theta1_rad = (theta1_arcmin * u.arcmin).to(u.radian).value
    theta2_rad = (theta2_arcmin * u.arcmin).to(u.radian).value
    nplanes = len(context.chis)
    zero_index = int(np.argmin(np.abs(lambdas)))
    deltas = np.zeros((nplanes, len(lambdas), 2), dtype=float)
    maximum_weight = np.max(np.abs(context.lensing_weight))
    negligible_weight = maximum_weight * 1e-07
    for plane in range(nplanes):
        print(f'[THEORY 2-cell] plane {plane + 1}/{nplanes}', end='\r')
        if np.abs(context.lensing_weight[plane]) <= negligible_weight:
            continue
        for index in range(zero_index + 1, len(lambdas)):
            deltas[plane, index] = solve_2cell_delta(context, lambdas[index], context.lensing_weight[plane], context.chis[plane], context.z_lens[plane], theta1_rad, theta2_rad, deltas[plane, index - 1])
        for index in range(zero_index - 1, -1, -1):
            deltas[plane, index] = solve_2cell_delta(context, lambdas[index], context.lensing_weight[plane], context.chis[plane], context.z_lens[plane], theta1_rad, theta2_rad, deltas[plane, index + 1])
    print()
    phi = np.zeros(len(lambdas), dtype=float)
    for index, lambda_value in enumerate(lambdas):
        for plane in range(nplanes):
            delta1, delta2 = deltas[plane, index]
            psi = psi_2cell_scalar(context, context.chis[plane], context.z_lens[plane], delta1, delta2, theta1_rad, theta2_rad)
            phi[index] += (lambda_value * context.lensing_weight[plane] * (delta1 - delta2) - psi) * context.dchis[plane]
    phi -= phi[zero_index]
    return phi
def compute_emulator_bin_edges(centres):
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 1 or centres.size < 2:
        raise ValueError('At least two one-dimensional bin centres are required.')
    mids = 0.5 * (centres[:-1] + centres[1:])
    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[1:-1] = mids
    edges[0] = 2.0 * centres[0] - mids[0]
    edges[-1] = 2.0 * centres[-1] - mids[-1]
    return edges


def load_emulator_l1_target(path, flip_sign=False):
    path = Path(path)
    obj = np.load(path, allow_pickle=True)
    data = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj
    if not isinstance(data, dict):
        raise TypeError(f'L1 target is not a dictionary: {path}')
    centres = np.asarray(data['kappa'], dtype=np.float64)
    pdf = np.asarray(data['pdf'], dtype=np.float64)
    if 'l1' in data:
        l1 = np.asarray(data['l1'], dtype=np.float64)
    elif 'l1_norm' in data:
        l1 = np.asarray(data['l1_norm'], dtype=np.float64)
    else:
        raise KeyError(f"Neither 'l1' nor 'l1_norm' exists in: {path}")
    order = np.argsort(centres)
    centres, pdf, l1 = (centres[order], pdf[order], l1[order])
    if flip_sign:
        centres = -centres[::-1]
        pdf = pdf[::-1]
        l1 = l1[::-1]
    return {'bincenters': centres, 'binedges': compute_emulator_bin_edges(centres), 'histogram': pdf, 'l1_norm': l1, 'source_path': str(path)}


def validate_emulation_scales(bands, coarse_arcmin):
    bands = [(float(a), float(b)) for a, b in bands]
    if len(bands) == 0:
        raise ValueError('EMULATION_L1_BANDS must contain at least one band.')
    for a, b in bands:
        if not (a > 0.0 and b > a):
            raise ValueError(f'Invalid L1 band: {(a, b)}')
    for (_, b0), (a1, _) in zip(bands[:-1], bands[1:]):
        if not np.isclose(b0, a1, rtol=0.0, atol=1e-12):
            raise ValueError(f'EMULATION_L1_BANDS must form a contiguous telescoping chain. Found a gap/overlap between {b0:g} and {a1:g} arcmin.')
    if not np.isclose(bands[-1][1], float(coarse_arcmin), rtol=0.0, atol=1e-12):
        raise ValueError('The final upper band edge must equal EMULATION_COARSE_ARCMIN.')
    return bands


def emulator_top_hat_beam(theta_arcmin, lmax):
    theta_rad = np.deg2rad(float(theta_arcmin) / 60.0)
    mu = np.cos(theta_rad)
    ell = np.arange(int(lmax) + 1, dtype=np.int64)
    beam = np.ones(int(lmax) + 1, dtype=np.float64)
    lp = ell[1:]
    p_minus = eval_legendre(lp - 1, mu)
    p_plus = eval_legendre(lp + 1, mu)
    beam[1:] = (p_minus - p_plus) / ((2.0 * lp + 1.0) * (1.0 - mu))
    beam[0] = 1.0
    return beam


def build_emulator_targets(bands, coarse_arcmin):
    targets = {}
    for a, b in bands:
        key = f'{a:g}-{b:g}'
        targets[key] = load_emulator_l1_target(theory_l1_path(a, b), flip_sign=True)
    targets[f'coarse-{coarse_arcmin:g}'] = load_emulator_l1_target(theory_l1_path(coarse=True), flip_sign=False)
    return targets


def decompose_map(solution, nside, lmax, bands, coarse_arcmin):
    bands = validate_emulation_scales(bands, coarse_arcmin)
    radii = sorted({bands[0][0], *[b for _, b in bands]})
    alm = hp.map2alm(np.asarray(solution, dtype=np.float64), lmax=int(lmax), iter=HARMONIC_ITER, pol=False)
    smooth = {}
    for theta in radii:
        beam = emulator_top_hat_beam(theta, lmax)
        smooth[theta] = hp.alm2map(hp.almxfl(alm, beam), nside=int(nside), lmax=int(lmax), pol=False)
    components = {'fine': np.asarray(solution, dtype=np.float64) - smooth[bands[0][0]]}
    for a, b in bands:
        components[f'{a:g}-{b:g}'] = smooth[a] - smooth[b]
    components[f'coarse-{coarse_arcmin:g}'] = smooth[float(coarse_arcmin)]
    del alm, smooth
    gc.collect()
    return components


def adjust_map_l1(input_image, target):
    """Project rank-ordered pixels onto target histogram/L1 constraints."""
    x = np.asarray(input_image, dtype=np.float64).copy()
    edges = np.asarray(target['binedges'], dtype=np.float64)
    pdf = np.asarray(target['histogram'], dtype=np.float64)
    target_l1 = np.asarray(target['l1_norm'], dtype=np.float64)
    widths = np.diff(edges)
    probs = np.maximum(pdf * widths, 0.0)
    prob_sum = float(np.sum(probs))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError('Invalid target histogram probability.')
    probs /= prob_sum
    order = np.argsort(x)
    sorted_x = x[order].copy()
    counts = np.floor(probs * x.size).astype(np.int64)
    counts[-1] += x.size - counts.sum()
    total_err = 0.0
    start = 0
    for j, n in enumerate(counts):
        n = int(n)
        end = start + n
        if n <= 0:
            start = end
            continue
        vals = sorted_x[start:end].copy()
        desired = float(target_l1[j] * widths[j] * x.size)
        current = float(np.sum(np.abs(vals)))
        delta = (desired - current) / float(n)
        positive = vals >= 0.0
        vals[positive] = np.maximum(vals[positive] + delta, 0.0)
        current = float(np.sum(np.abs(vals)))
        negative = ~positive
        n_negative = int(np.count_nonzero(negative))
        if n_negative > 0:
            vals[negative] = np.minimum(vals[negative] - (desired - current) / float(n_negative), 0.0)
        vals = np.clip(vals, edges[j], edges[j + 1])
        achieved = float(np.sum(np.abs(vals)))
        total_err += abs(desired - achieved)
        sorted_x[start:end] = vals
        start = end
    output = np.empty_like(x)
    output[order] = sorted_x
    norm = max(float(np.sum(target_l1 * widths)), 1e-30) * x.size
    return (output, float(total_err / norm))


def adjust_cls(input_map, nside, lmax, target_cls):
    alm = hp.map2alm(np.asarray(input_map, dtype=np.float64), lmax=int(lmax), iter=HARMONIC_ITER, pol=False)
    current_cls = hp.alm2cl(alm, lmax=int(lmax))
    factor = np.sqrt(np.maximum(target_cls, 0.0) / np.maximum(current_cls, 1e-30))
    factor[:2] = 0.0
    adjusted_map = hp.alm2map(hp.almxfl(alm, factor), nside=int(nside), lmax=int(lmax), pol=False)
    del alm, current_cls, factor
    gc.collect()
    return adjusted_map


def emulation_map_path(realisation_index):
    return EMULATION_OUTPUT_DIR / f'emu_{source_token()}_Nside{EMULATION_NSIDE}_lmax{EMULATION_LMAX}_{int(realisation_index):02d}.npy'


def emulation_history_path(realisation_index):
    return EMULATION_OUTPUT_DIR / f'emu_{source_token()}_Nside{EMULATION_NSIDE}_lmax{EMULATION_LMAX}_{int(realisation_index):02d}_l1_history.npz'
def aggregate_wavelet_key_prefix(theta1, theta2):
    """NPZ key prefix for one wavelet band."""
    theta1_token = f'{float(theta1):g}'.replace('.', '_')
    theta2_token = f'{float(theta2):g}'.replace('.', '_')
    return f'wavelet_{theta1_token}_{theta2_token}'


def aggregate_coarse_key_prefix(theta):
    """NPZ key prefix for the coarse component."""
    theta_token = f'{float(theta):g}'.replace('.', '_')
    return f'coarse_{theta_token}'


def load_aggregate_emulator_target(npz_data, key_prefix, source_path):
    """Convert one aggregate-NPZ component to the emulator target format."""
    required_keys = {f'{key_prefix}_kappa', f'{key_prefix}_pdf', f'{key_prefix}_l1'}
    missing_keys = required_keys - set(npz_data.files)
    if missing_keys:
        raise KeyError(f'Missing target keys in {source_path}: ' + ', '.join(sorted(missing_keys)))
    centres = np.asarray(npz_data[f'{key_prefix}_kappa'], dtype=np.float64)
    pdf = np.asarray(npz_data[f'{key_prefix}_pdf'], dtype=np.float64)
    l1 = np.asarray(npz_data[f'{key_prefix}_l1'], dtype=np.float64)
    if not (centres.ndim == 1 and pdf.ndim == 1 and (l1.ndim == 1) and (centres.size == pdf.size == l1.size) and (centres.size >= 2)):
        raise ValueError(f'Invalid target-array shapes for {key_prefix} in {source_path}.')
    finite = np.isfinite(centres) & np.isfinite(pdf) & np.isfinite(l1)
    centres = centres[finite]
    pdf = pdf[finite]
    l1 = l1[finite]
    order = np.argsort(centres)
    centres = centres[order]
    pdf = pdf[order]
    l1 = l1[order]
    centres, unique_indices = np.unique(centres, return_index=True)
    pdf = pdf[unique_indices]
    l1 = l1[unique_indices]
    if centres.size < 2:
        raise ValueError(f'Too few finite unique kappa values for {key_prefix} in {source_path}.')
    return {'bincenters': centres, 'binedges': compute_emulator_bin_edges(centres), 'histogram': pdf, 'l1_norm': l1, 'source_path': str(source_path)}


def build_aggregate_emulator_targets(path, bands, coarse_arcmin):
    """Load all emulator L1 targets from one aggregate NPZ file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Aggregate theoretical L1 file not found:\n{path}')
    targets = {}
    with np.load(path, allow_pickle=False) as npz_data:
        for theta1, theta2 in bands:
            name = f'{theta1:g}-{theta2:g}'
            key_prefix = aggregate_wavelet_key_prefix(theta1, theta2)
            targets[name] = load_aggregate_emulator_target(npz_data, key_prefix, path)
        coarse_name = f'coarse-{coarse_arcmin:g}'
        coarse_key_prefix = aggregate_coarse_key_prefix(coarse_arcmin)
        targets[coarse_name] = load_aggregate_emulator_target(npz_data, coarse_key_prefix, path)
    return targets

def _configure_source() -> None:
    global SOURCE_CONFIG, SOURCE_MODE, SOURCE_Z_GRID, SOURCE_NZ
    global SOURCE_EFFECTIVE_REDSHIFT, SOURCE_REDSHIFT_STD, SOURCE_MEDIAN_REDSHIFT, SOURCE_PEAK_REDSHIFT
    global THEORY_ALL_SCALE_NPZ
    SOURCE_CONFIG = configure_source_population(NZ_FILE, SOURCE_PLANE_REDSHIFT)
    SOURCE_MODE = SOURCE_CONFIG["mode"]
    SOURCE_Z_GRID = SOURCE_CONFIG["z"]
    SOURCE_NZ = SOURCE_CONFIG["nz"]
    SOURCE_EFFECTIVE_REDSHIFT = SOURCE_CONFIG["effective_redshift"]
    SOURCE_REDSHIFT_STD = SOURCE_CONFIG["redshift_std"]
    SOURCE_MEDIAN_REDSHIFT = SOURCE_CONFIG["median_redshift"]
    SOURCE_PEAK_REDSHIFT = SOURCE_CONFIG["peak_redshift"]
    THEORY_ALL_SCALE_NPZ = L1_DIR / f"theory_{source_token()}_all_scale.npz"


def compute_theoretical_power_spectra():
    global SOURCE_Z, ELL_MIN, ELL_MAX, N_GLOBAL_BINS
    global C_KM_S, OMEGA_M, DELTA_C, RHO_CRIT_0
    # The power-spectrum helper functions below are module-level functions that
    # share the production CAMB and halo-model state created in this routine.
    # Keep these names explicitly global so the packaged module preserves the
    # same execution model as the original notebook implementation.
    global results, pk_lin, pk_nl
    global mass_h, mass_phys, lnM, wlnM, rho_m0, qM, qmin, sigma_k
    global theta_nodes, theta_weights, wp

    # ============================================================
    # THEORETICAL POWER-SPECTRUM PRODUCT
    # ============================================================

    THEORY_PS_PATH = TARGET_POWER_NPZ

    SOURCE_Z = float(SOURCE_EFFECTIVE_REDSHIFT)
    NSIDE = int(THEORY_NSIDE)

    ELL_MIN = int(THEORY_ELL_MIN)
    ELL_MAX = int(THEORY_ELL_MAX)

    N_GLOBAL_BINS = int(N_POWER_SPECTRUM_BINS)

    COV_ELL_MIN = float(ELL_MIN)
    COV_ELL_MAX = float(ELL_MAX)

    OMEGA_M = OMEGA_B + OMEGA_CDM

    C_KM_S = 299792.458
    RHO_CRIT_0 = 2.77536627e11
    DELTA_C = 1.68647019984


    # ============================================================
    # Required fields in the saved theoretical product
    # ============================================================

    required_theory_keys = {
        "ells",
        "target_mean",
        "all_cls",
        "theory_target_mean_binned",
        "cov_gaussian",
        "cov_cng",
        "cov_ssc",
        "cov_total",
        "correlation_total",
        "correlation_bin_edges",
        "binning_matrix",
    }


    # ============================================================
    # Load or generate the theoretical power-spectrum product
    # ============================================================

    if THEORY_PS_PATH.exists():

        print(
            f"[LOAD] theory power-spectrum product: "
            f"{THEORY_PS_PATH}"
        )

        theory_product = load_npz_product(
            THEORY_PS_PATH
        )

        upgraded = False


        # --------------------------------------------------------
        # Restore binning information for products written by
        # earlier notebook versions.
        # --------------------------------------------------------

        if "binning_matrix" not in theory_product:

            if "ells" not in theory_product:
                raise KeyError(
                    "Cannot reconstruct binning_matrix because "
                    "'ells' is missing from the saved theory product."
                )

            saved_ells = np.asarray(
                theory_product["ells"],
                dtype=int,
            )

            (
                reconstructed_bin_edges,
                reconstructed_B,
                reconstructed_ell_eff,
                reconstructed_bin_lo,
                reconstructed_bin_hi,
                reconstructed_mode_count,
            ) = build_global_binning(
                saved_ells
            )

            theory_product["binning_matrix"] = np.asarray(
                reconstructed_B,
                dtype=np.float64,
            )

            upgraded = True

            print(
                "[UPGRADE] reconstructed binning_matrix"
            )


        # --------------------------------------------------------
        # Restore total correlation matrix from total covariance.
        # --------------------------------------------------------

        if "correlation_total" not in theory_product:

            if "cov_total" not in theory_product:
                raise KeyError(
                    "Cannot reconstruct correlation_total because "
                    "'cov_total' is missing from the saved theory product."
                )

            theory_product["correlation_total"] = np.asarray(
                cov_to_corr(
                    np.asarray(
                        theory_product["cov_total"],
                        dtype=float,
                    )
                ),
                dtype=np.float64,
            )

            upgraded = True

            print(
                "[UPGRADE] reconstructed correlation_total"
            )


        # --------------------------------------------------------
        # Restore additional bin metadata when possible.
        # --------------------------------------------------------

        if (
            "ells" in theory_product
            and (
                "correlation_bin_edges" not in theory_product
                or "correlation_bin_ell_min" not in theory_product
                or "correlation_bin_ell_max" not in theory_product
                or "correlation_bin_ell_eff" not in theory_product
                or "correlation_bin_mode_count" not in theory_product
            )
        ):

            saved_ells = np.asarray(
                theory_product["ells"],
                dtype=int,
            )

            (
                reconstructed_bin_edges,
                reconstructed_B,
                reconstructed_ell_eff,
                reconstructed_bin_lo,
                reconstructed_bin_hi,
                reconstructed_mode_count,
            ) = build_global_binning(
                saved_ells
            )

            if "correlation_bin_edges" not in theory_product:
                theory_product["correlation_bin_edges"] = np.asarray(
                    reconstructed_bin_edges,
                    dtype=np.int64,
                )
                upgraded = True

            if "correlation_bin_ell_min" not in theory_product:
                theory_product["correlation_bin_ell_min"] = np.asarray(
                    reconstructed_bin_lo,
                    dtype=np.int64,
                )
                upgraded = True

            if "correlation_bin_ell_max" not in theory_product:
                theory_product["correlation_bin_ell_max"] = np.asarray(
                    reconstructed_bin_hi,
                    dtype=np.int64,
                )
                upgraded = True

            if "correlation_bin_ell_eff" not in theory_product:
                theory_product["correlation_bin_ell_eff"] = np.asarray(
                    reconstructed_ell_eff,
                    dtype=np.float64,
                )
                upgraded = True

            if "correlation_bin_mode_count" not in theory_product:
                theory_product["correlation_bin_mode_count"] = np.asarray(
                    reconstructed_mode_count,
                    dtype=np.float64,
                )
                upgraded = True


        # --------------------------------------------------------
        # Validate all non-reconstructable required fields.
        # --------------------------------------------------------

        missing = required_theory_keys.difference(
            theory_product
        )

        if missing:

            raise KeyError(
                "The saved theory product is missing "
                "non-reconstructable required fields: "
                + ", ".join(
                    sorted(missing)
                )
            )


        # --------------------------------------------------------
        # Save the upgraded product once so future runs can load
        # it directly without reconstruction.
        # --------------------------------------------------------

        if upgraded:

            save_theory_power_product(
                THEORY_PS_PATH,
                **theory_product,
            )

            print(
                f"[SAVE] upgraded theory power-spectrum product: "
                f"{THEORY_PS_PATH}"
            )


    # ============================================================
    # Compute the theoretical product when no saved file exists
    # ============================================================

    else:

        print(
            f"[COMPUTE] theory power-spectrum product: "
            f"{THEORY_PS_PATH}"
        )


        # ========================================================
        # Multipole grid and covariance binning
        # ========================================================

        ells = np.arange(
            ELL_MIN,
            ELL_MAX + 1,
            dtype=int,
        )

        (
            bin_edges,
            B_global,
            ell_eff_global,
            bin_lo_global,
            bin_hi_global,
            mode_count_global,
        ) = build_global_binning(
            ells
        )

        sel_cov = (
            (ell_eff_global >= COV_ELL_MIN)
            & (ell_eff_global <= COV_ELL_MAX)
        )

        cov_bin_idx = np.where(
            sel_cov
        )[0]

        ell_eff = ell_eff_global[
            sel_cov
        ]

        bin_lo = bin_lo_global[
            sel_cov
        ]

        bin_hi = bin_hi_global[
            sel_cov
        ]

        N_COV = ell_eff.size

        print(
            "\nSelected covariance bins:"
        )

        for i, (le, lo, hi) in enumerate(
            zip(
                ell_eff,
                bin_lo,
                bin_hi,
            )
        ):
            print(
                f"  {i:2d}: "
                f"ell_eff={le:8.2f}   "
                f"[{lo},{hi}]"
            )


        # ========================================================
        # Match A_s to the requested sigma8
        # ========================================================

        print(
            "\n[1/7] Matching A_s to sigma8 with CAMB..."
        )

        As0 = 2.10

        r0 = camb.get_results(
            make_camb_params(
                As0,
                kmax=30.0,
            )
        )

        s8_0 = float(
            np.asarray(
                r0.get_sigma8()
            ).reshape(-1)[-1]
        )

        As_match = (
            As0
            * (SIGMA8_TARGET / s8_0) ** 2
        )

        r1 = camb.get_results(
            make_camb_params(
                As_match,
                kmax=30.0,
            )
        )

        s8_1 = float(
            np.asarray(
                r1.get_sigma8()
            ).reshape(-1)[-1]
        )

        As_match *= (
            SIGMA8_TARGET / s8_1
        ) ** 2

        print(
            f"  initial sigma8={s8_0:.10f}; "
            f"matched As={As_match:.10f} x1e-9"
        )

        del r0, r1

        gc.collect()


        # ========================================================
        # Build the CAMB matter-power backend
        # ========================================================

        print(
            "[2/7] Building production CAMB matter-power backend..."
        )

        results = camb.get_results(
            make_camb_params(
                As_match,
                kmax=CAMB_KMAX_MPC,
            )
        )

        pk_lin = results.get_matter_power_interpolator(
            nonlinear=False,
            hubble_units=False,
            k_hunit=False,
            log_interp=True,
            extrap_kmax=PROFILE_KMAX_MPC,
            silent=True,
        )

        pk_nl = results.get_matter_power_interpolator(
            nonlinear=True,
            hubble_units=False,
            k_hunit=False,
            log_interp=True,
            extrap_kmax=PROFILE_KMAX_MPC,
            silent=True,
        )


        # ========================================================
        # Compute the nonlinear mean C_ell
        # ========================================================

        print(
            "[3/7] Computing exact-source-plane nonlinear mean C_ell..."
        )

        z_mean = np.linspace(
            1.0e-4,
            SOURCE_Z,
            N_MEAN_Z,
        )

        (
            chi_mean,
            W_mean,
            chi_s,
        ) = source_kernel(
            z_mean
        )

        wchi_mean = trap_weights(
            chi_mean
        )

        cl_unpix = np.zeros(
            ells.size,
            dtype=float,
        )

        for (
            zz,
            cc,
            ww,
            dw,
        ) in zip(
            z_mean,
            chi_mean,
            W_mean,
            wchi_mean,
        ):

            if (
                cc <= 0.0
                or ww == 0.0
            ):
                continue

            kk = (
                ells.astype(float) + 0.5
            ) / cc

            good = (
                (kk >= 1.0e-5)
                & (kk <= PROFILE_KMAX_MPC)
            )

            if np.any(good):

                pp = P_nl(
                    float(zz),
                    kk[good],
                )

                cl_unpix[good] += (
                    dw
                    * (ww ** 2 / cc ** 2)
                    * pp
                )

        if (
            not np.all(
                np.isfinite(cl_unpix)
            )
            or np.any(
                cl_unpix <= 0.0
            )
        ):
            raise RuntimeError(
                "Invalid theoretical C_ell."
            )


        # ========================================================
        # Pixel window and Gaussian covariance
        # ========================================================

        pixwin_full = np.asarray(
            hp.pixwin(
                NSIDE,
                lmax=ELL_MAX,
                pol=False,
            ),
            dtype=float,
        )

        pixwin = pixwin_full[
            ells
        ]

        pix2 = pixwin ** 2

        cl_pix = (
            cl_unpix
            * pix2
        )

        cl_bin_global = (
            B_global
            @ cl_pix
        )

        cl_bin_unpix_global = (
            B_global
            @ cl_unpix
        )

        varG_unbinned = (
            2.0
            * (cl_pix + NOISE_LEVEL) ** 2
            / (
                (2.0 * ells + 1.0)
                * F_SKY
            )
        )

        covG_global = (
            B_global
            * varG_unbinned[
                None,
                :
            ]
        ) @ B_global.T

        cov_G = covG_global[
            np.ix_(
                cov_bin_idx,
                cov_bin_idx,
            )
        ]

        pix2_eff = np.interp(
            ell_eff,
            ells.astype(float),
            pix2,
        )


        # ========================================================
        # Halo-model grids
        # ========================================================

        mass_h = np.geomspace(
            10.0 ** LOG10_M_MIN,
            10.0 ** LOG10_M_MAX,
            N_MASS,
        )

        lnM = np.log(
            mass_h
        )

        wlnM = trap_weights(
            lnM
        )

        mass_phys = (
            mass_h / H
        )

        rho_m0 = (
            RHO_CRIT_0
            * H ** 2
            * OMEGA_M
        )

        qM = (
            mass_phys
            / rho_m0
        )

        qmin = qM[0]

        sigma_k = np.geomspace(
            SIGMA_K_MIN,
            SIGMA_K_MAX,
            SIGMA_NK,
        )

        xg, wg = leggauss(
            N_ANGLE
        )

        theta_nodes = (
            0.5
            * np.pi
            * (xg + 1.0)
        )

        theta_weights = (
            0.5
            * wg
        )


        # ========================================================
        # cNG and halo-response fields
        # ========================================================

        print(
            "[4/7] Computing 1-4 halo cNG and halo-model SSC response fields..."
        )

        z_halo = np.linspace(
            1.0e-4,
            SOURCE_Z,
            N_HALO_Z,
        )

        (
            chi_halo,
            W_halo,
            _,
        ) = source_kernel(
            z_halo
        )

        wchi_halo = trap_weights(
            chi_halo
        )

        area_sr = (
            4.0
            * np.pi
            * F_SKY
        )

        cov_1h_u = np.zeros(
            (N_COV, N_COV)
        )

        cov_2h22_u = np.zeros_like(
            cov_1h_u
        )

        cov_2h13_u = np.zeros_like(
            cov_1h_u
        )

        cov_3h_u = np.zeros_like(
            cov_1h_u
        )

        cov_4h_u = np.zeros_like(
            cov_1h_u
        )

        PR1 = np.zeros(
            (N_COV, N_HALO_Z)
        )

        PRK = np.zeros(
            (N_COV, N_HALO_Z)
        )

        Dgrowth = np.zeros(
            N_HALO_Z
        )

        P0_growth = float(
            P_lin(
                0.0,
                np.array(
                    [1.0e-3]
                ),
            )[0]
        )

        for iz, (
            zz,
            cc,
            ww,
            dw,
        ) in enumerate(
            zip(
                z_halo,
                chi_halo,
                W_halo,
                wchi_halo,
            )
        ):

            if (
                iz % 10 == 0
                or iz == N_HALO_Z - 1
            ):
                print(
                    f"  halo z "
                    f"{iz + 1:02d}/{N_HALO_Z}, "
                    f"z={zz:.4f}"
                )

            if (
                cc <= 0.0
                or ww == 0.0
            ):
                continue

            k0 = (
                ell_eff + 0.5
            ) / cc

            em = np.exp(
                -RESPONSE_DLOGK
            )

            ep = np.exp(
                +RESPONSE_DLOGK
            )

            kall = np.concatenate(
                [
                    k0 * em,
                    k0,
                    k0 * ep,
                ]
            )

            state = halo_state(
                float(zz)
            )

            hq_all = halo_integrals(
                float(zz),
                kall,
                state=state,
                need_matrices=True,
                need_conc_derivs=True,
            )

            n = N_COV

            slm = slice(
                0,
                n,
            )

            sl0 = slice(
                n,
                2 * n,
            )

            slp = slice(
                2 * n,
                3 * n,
            )

            I11 = hq_all[
                "I11"
            ][sl0]

            I02 = hq_all[
                "I02"
            ][sl0]

            I11c = hq_all[
                "I11c"
            ][sl0]

            I02c = hq_all[
                "I02c"
            ][sl0]

            I12 = hq_all[
                "I12"
            ][sl0, sl0]

            I13 = hq_all[
                "I13"
            ][sl0, sl0]

            I04 = hq_all[
                "I04"
            ][sl0, sl0]

            Pl = safe_Plin(
                float(zz),
                k0,
            )

            Pm = safe_Plin(
                float(zz),
                k0 * em,
            )

            Pp = safe_Plin(
                float(zz),
                k0 * ep,
            )

            PHM = (
                I11 ** 2
                * Pl
                + I02
            )

            PHM_m = (
                hq_all["I11"][slm] ** 2
                * Pm
                + hq_all["I02"][slm]
            )

            PHM_p = (
                hq_all["I11"][slp] ** 2
                * Pp
                + hq_all["I02"][slp]
            )

            dlnPHM = (
                np.log(
                    np.maximum(
                        PHM_p,
                        1.0e-300,
                    )
                )
                - np.log(
                    np.maximum(
                        PHM_m,
                        1.0e-300,
                    )
                )
            ) / (
                2.0
                * RESPONSE_DLOGK
            )

            dlnPl = (
                np.log(
                    np.maximum(
                        Pp,
                        1.0e-300,
                    )
                )
                - np.log(
                    np.maximum(
                        Pm,
                        1.0e-300,
                    )
                )
            ) / (
                2.0
                * RESPONSE_DLOGK
            )

            PR1[:, iz] = (
                (
                    47.0 / 21.0
                    - dlnPl / 3.0
                )
                * Pl
                * I11 ** 2
                + np.diag(I12)
            )

            GK_times_PHM = (
                (8.0 / 7.0)
                * I11 ** 2
                * Pl
                + C_K_TIDAL
                * (
                    2.0
                    * I11c
                    * I11
                    * Pl
                    + I02c
                )
            )

            PRK[:, iz] = (
                GK_times_PHM
                - PHM
                * dlnPHM
            )

            (
                P_iso,
                Bpt,
                Tpt,
            ) = isotropic_pt_pieces(
                float(zz),
                k0,
            )

            i1r = I11[
                None,
                :
            ]

            i1c = I11[
                :,
                None,
            ]

            T1 = I04

            T22 = (
                2.0
                * P_iso
                * I12 ** 2
            )

            T13 = (
                2.0
                * (
                    Pl[None, :]
                    * i1r
                    * I13
                    + Pl[:, None]
                    * i1c
                    * I13.T
                )
            )

            T3 = (
                4.0
                * Bpt
                * i1r
                * i1c
                * I12
            )

            T4 = (
                i1r ** 2
                * i1c ** 2
                * Tpt
            )

            proj = (
                dw
                * ww ** 4
                / cc ** 6
                / area_sr
            )

            cov_1h_u += (
                proj
                * T1
            )

            cov_2h22_u += (
                proj
                * T22
            )

            cov_2h13_u += (
                proj
                * T13
            )

            cov_3h_u += (
                proj
                * T3
            )

            cov_4h_u += (
                proj
                * T4
            )

            Dgrowth[iz] = np.sqrt(
                max(
                    float(
                        P_lin(
                            float(zz),
                            np.array(
                                [1.0e-3]
                            ),
                        )[0]
                    )
                    / P0_growth,
                    0.0,
                )
            )


        for C in (
            cov_1h_u,
            cov_2h22_u,
            cov_2h13_u,
            cov_3h_u,
            cov_4h_u,
        ):
            C[:] = (
                0.5
                * (
                    C
                    + C.T
                )
            )

        cov_cng_u = (
            cov_1h_u
            + cov_2h22_u
            + cov_2h13_u
            + cov_3h_u
            + cov_4h_u
        )

        pix_outer = np.outer(
            pix2_eff,
            pix2_eff,
        )

        cov_1h = (
            cov_1h_u
            * pix_outer
        )

        cov_2h22 = (
            cov_2h22_u
            * pix_outer
        )

        cov_2h13 = (
            cov_2h13_u
            * pix_outer
        )

        cov_3h = (
            cov_3h_u
            * pix_outer
        )

        cov_4h = (
            cov_4h_u
            * pix_outer
        )

        cov_cng = (
            cov_cng_u
            * pix_outer
        )


        # ========================================================
        # Full-sky SSC
        # ========================================================

        print(
            "[5/7] Computing full-sky curved-sky L=0 SSC..."
        )

        p_long = np.geomspace(
            LONG_K_MIN,
            LONG_K_MAX,
            N_LONG,
        )

        lnp = np.log(
            p_long
        )

        wlnp = trap_weights(
            lnp
        )

        P_long0 = safe_Plin(
            0.0,
            p_long,
        )

        F_density = np.zeros(
            (N_COV, N_LONG)
        )

        F_total = np.zeros(
            (N_COV, N_LONG)
        )

        for ip, pp in enumerate(
            p_long
        ):

            x = (
                pp
                * chi_halo
            )

            j0 = spherical_jn(
                0,
                x,
            )

            j0dd = j0_second(
                x
            )

            common = (
                wchi_halo
                * W_halo ** 2
                / np.maximum(
                    chi_halo,
                    1.0e-30,
                ) ** 2
                * Dgrowth
            )

            F_density[:, ip] = np.sum(
                PR1
                * (
                    common
                    * j0
                )[None, :],
                axis=1,
            )

            tidal_kernel = (
                j0 / 6.0
                + 0.5
                * j0dd
            )

            F_total[:, ip] = np.sum(
                PR1
                * (
                    common
                    * j0
                )[None, :]
                + PRK
                * (
                    common
                    * tidal_kernel
                )[None, :],
                axis=1,
            )

        wp = (
            (2.0 / np.pi)
            * wlnp
            * p_long ** 3
            * P_long0
        )

        cov_ssc_density_u = ssc_from_F(
            F_density
        )

        cov_ssc_u = ssc_from_F(
            F_total
        )

        cov_ssc_density = (
            cov_ssc_density_u
            * pix_outer
        )

        cov_ssc = (
            cov_ssc_u
            * pix_outer
        )


        # ========================================================
        # Total covariance
        # ========================================================

        cov_G_cNG = (
            cov_G
            + cov_cng
        )

        cov_theory_raw = (
            cov_G
            + cov_cng
            + cov_ssc
        )

        cov_theory = project_psd(
            cov_theory_raw,
            PSD_RELATIVE_FLOOR,
        )

        R_theory = cov_to_corr(
            cov_theory
        )

        psd_change = (
            np.linalg.norm(
                cov_theory
                - cov_theory_raw
            )
            / max(
                np.linalg.norm(
                    cov_theory_raw
                ),
                1.0e-300,
            )
        )


        # ========================================================
        # Generate theoretical realisations
        # ========================================================

        print(
            "[6/7] Generating 100 theory power-spectrum realisations..."
        )

        (
            theory_binned_samples,
            chisq_df,
            chisq_scale,
            sampling_target_corr,
            sampling_latent_corr,
        ) = chisquare_copula_samples(
            cl_bin_global,
            cov_theory,
            n_samples=N_THEORY_POWER_SPECTRA,
            seed=THEORY_RANDOM_SEED,
        )

        bin_ratio = (
            theory_binned_samples
            / cl_bin_global[
                None,
                :
            ]
        )

        theory_all_cls = np.empty(
            (
                N_THEORY_POWER_SPECTRA,
                ells.size,
            ),
            dtype=float,
        )

        assigned = np.zeros(
            ells.size,
            dtype=bool,
        )

        for b in range(
            N_GLOBAL_BINS
        ):

            mask = (
                B_global[b]
                > 0.0
            )

            theory_all_cls[
                :,
                mask,
            ] = (
                cl_pix[
                    None,
                    mask,
                ]
                * bin_ratio[
                    :,
                    b,
                ][
                    :,
                    None,
                ]
            )

            assigned[
                mask
            ] = True

        if not np.all(
            assigned
        ):
            raise RuntimeError(
                "Some ell values were not assigned to a theory bin."
            )

        theory_binned_from_full = (
            theory_all_cls
            @ B_global.T
        )

        theory_sample_cov = np.cov(
            theory_binned_from_full,
            rowvar=False,
            ddof=1,
        )

        theory_sample_corr = cov_to_corr(
            theory_sample_cov
        )

        theory_sample_mean = np.mean(
            theory_binned_from_full,
            axis=0,
        )

        print(
            f"  theory_all_cls shape = "
            f"{theory_all_cls.shape}"
        )

        print(
            f"  total covariance shape = "
            f"{cov_theory.shape}"
        )

        print(
            f"  PSD projection relative change = "
            f"{psd_change:.3e}"
        )


        # ========================================================
        # Save the complete theoretical product
        # ========================================================

        theory_product = {

            "ells":
                np.asarray(
                    ells,
                    dtype=np.int64,
                ),

            "target_mean":
                np.asarray(
                    cl_pix,
                    dtype=np.float64,
                ),

            "target_mean_unpixelized":
                np.asarray(
                    cl_unpix,
                    dtype=np.float64,
                ),

            "all_cls":
                np.asarray(
                    theory_all_cls,
                    dtype=np.float64,
                ),

            "theory_target_mean_binned":
                np.asarray(
                    cl_bin_global,
                    dtype=np.float64,
                ),

            "sampled_bandpowers":
                np.asarray(
                    theory_binned_samples,
                    dtype=np.float64,
                ),

            "sample_covariance":
                np.asarray(
                    theory_sample_cov,
                    dtype=np.float64,
                ),

            "sample_correlation":
                np.asarray(
                    theory_sample_corr,
                    dtype=np.float64,
                ),

            "cov_gaussian":
                np.asarray(
                    cov_G,
                    dtype=np.float64,
                ),

            "cov_1h":
                np.asarray(
                    cov_1h,
                    dtype=np.float64,
                ),

            "cov_2h22":
                np.asarray(
                    cov_2h22,
                    dtype=np.float64,
                ),

            "cov_2h13":
                np.asarray(
                    cov_2h13,
                    dtype=np.float64,
                ),

            "cov_3h":
                np.asarray(
                    cov_3h,
                    dtype=np.float64,
                ),

            "cov_4h":
                np.asarray(
                    cov_4h,
                    dtype=np.float64,
                ),

            "cov_cng":
                np.asarray(
                    cov_cng,
                    dtype=np.float64,
                ),

            "cov_ssc":
                np.asarray(
                    cov_ssc,
                    dtype=np.float64,
                ),

            "cov_total":
                np.asarray(
                    cov_theory,
                    dtype=np.float64,
                ),

            "correlation_total":
                np.asarray(
                    R_theory,
                    dtype=np.float64,
                ),

            "correlation_bin_edges":
                np.asarray(
                    bin_edges,
                    dtype=np.int64,
                ),

            "correlation_bin_ell_min":
                np.asarray(
                    bin_lo_global,
                    dtype=np.int64,
                ),

            "correlation_bin_ell_max":
                np.asarray(
                    bin_hi_global,
                    dtype=np.int64,
                ),

            "correlation_bin_ell_eff":
                np.asarray(
                    ell_eff_global,
                    dtype=np.float64,
                ),

            "correlation_bin_mode_count":
                np.asarray(
                    mode_count_global,
                    dtype=np.float64,
                ),

            "binning_matrix":
                np.asarray(
                    B_global,
                    dtype=np.float64,
                ),

            "nside":
                np.int64(
                    NSIDE
                ),

            "ell_min":
                np.int64(
                    ELL_MIN
                ),

            "ell_max":
                np.int64(
                    ELL_MAX
                ),

            "n_bins":
                np.int64(
                    N_GLOBAL_BINS
                ),

            "n_realisations":
                np.int64(
                    N_THEORY_POWER_SPECTRA
                ),

            "sampling_seed":
                np.int64(
                    THEORY_RANDOM_SEED
                ),

            "sampling_method":
                np.asarray(
                    "gaussian_copula_scaled_chi_square"
                ),

            "source_redshift":
                np.float64(
                    SOURCE_Z
                ),

            "f_sky":
                np.float64(
                    F_SKY
                ),

            "noise_level":
                np.float64(
                    NOISE_LEVEL
                ),

            "sigma8":
                np.float64(
                    SIGMA8_TARGET
                ),

            "halofit_version":
                np.asarray(
                    HALOFIT_VERSION
                ),

        }

        save_theory_power_product(
            THEORY_PS_PATH,
            **theory_product,
        )

        print(
            f"[SAVE] theory power-spectrum product: "
            f"{THEORY_PS_PATH}"
        )


    # ============================================================
    # ============================================================

    return theory_product


def compute_theoretical_wavelet_l1():
    """Compute and save the theoretical wavelet L1-norm targets."""
    ctx = ThreeScaleTheoryContext.build()
    products = {}
    for theta1, theta2 in EMULATION_L1_BANDS:
        prefix = aggregate_wavelet_key_prefix(theta1, theta2)
        product = wavelet_theory_product(ctx, theta1, theta2, prefix)
        products[f"{prefix}_kappa"] = np.asarray(product["kappa"], dtype=np.float64)
        products[f"{prefix}_pdf"] = np.asarray(product["pdf"], dtype=np.float64)
        products[f"{prefix}_l1"] = np.asarray(product["l1"], dtype=np.float64)
    prefix = aggregate_coarse_key_prefix(EMULATION_COARSE_ARCMIN)
    product = coarse_theory_product(ctx, EMULATION_COARSE_ARCMIN)
    products[f"{prefix}_kappa"] = np.asarray(product["kappa"], dtype=np.float64)
    products[f"{prefix}_pdf"] = np.asarray(product["pdf"], dtype=np.float64)
    products[f"{prefix}_l1"] = np.asarray(product["l1"], dtype=np.float64)
    np.savez_compressed(THEORY_ALL_SCALE_NPZ, **products)
    return products


def generate_emulated_maps(config: WelcomeConfig | None = None):
    """Generate emulated maps, with process-safe configuration for parallel workers."""
    worker_config = config if config is not None else ACTIVE_CONFIG
    if worker_config is None:
        raise RuntimeError("No active WELCOME configuration. Pass WelcomeConfig to generate_emulated_maps().")
    _apply_config(worker_config)
    _configure_source()
    BANDS = validate_emulation_scales(EMULATION_L1_BANDS, EMULATION_COARSE_ARCMIN)
    TARGET_L1 = build_aggregate_emulator_targets(THEORY_ALL_SCALE_NPZ, BANDS, EMULATION_COARSE_ARCMIN)
    REALISATION_INDICES = np.arange(REALISATION_START, REALISATION_END + 1, dtype=np.int64)
    with np.load(TARGET_POWER_NPZ, allow_pickle=False) as f:
        target_ells = np.asarray(f['ells'], dtype=np.int64)
        target_all_cls = np.asarray(f['all_cls'], dtype=np.float64)
    required_ells = np.arange(2, EMULATION_LMAX + 1, dtype=np.int64)
    pos = np.searchsorted(target_ells, required_ells)
    if np.any(pos >= target_ells.size) or not np.array_equal(target_ells[pos], required_ells):
        missing = np.setdiff1d(required_ells, target_ells)
        raise ValueError(f'Target C_ell does not cover all required multipoles. First missing: {missing[:20]}')
    if REALISATION_END >= target_all_cls.shape[0]:
        raise ValueError(f'Requested realisation {REALISATION_END}, but only {target_all_cls.shape[0]} C_ell realisations exist.')
    print()
    print('=' * 96)
    print('EMULATION START')
    print('=' * 96)
    print('NSIDE                 :', EMULATION_NSIDE)
    print('lmax                  :', EMULATION_LMAX)
    print('harmonic iter         :', HARMONIC_ITER)
    print('realisations          :', REALISATION_INDICES.tolist())
    print('iterations            :', N_ITERATIONS)
    print('n_jobs                :', N_JOBS)
    print('L1 bands              :', [f'{a:g}-{b:g}' for a, b in BANDS])
    print('coarse                :', f'{EMULATION_COARSE_ARCMIN:g} arcmin')
    print('fine residual         :', f'< {BANDS[0][0]:g} arcmin preserved')
    print('output                :', EMULATION_OUTPUT_DIR)
    print('=' * 96)
    def run_one_emulation(realisation_index):
        # Windows/loky workers start in fresh Python processes, so initialise
        # module state explicitly from the serialisable configuration object.
        _apply_config(worker_config)
        _configure_source()
        realisation_index = int(realisation_index)
        map_path = emulation_map_path(realisation_index)
        history_path = emulation_history_path(realisation_index)
        scale_names = [f'{a:g}-{b:g}' for a, b in BANDS] + [f'coarse-{EMULATION_COARSE_ARCMIN:g}']
        l1_target_paths = [THEORY_ALL_SCALE_NPZ]
        l1_target_mtime_ns = np.asarray([path.stat().st_mtime_ns for path in l1_target_paths], dtype=np.int64)
        target_power_mtime_ns = np.int64(TARGET_POWER_NPZ.stat().st_mtime_ns)
        expected_random_seed = int(RANDOM_SEED + realisation_index)
        if not OVERWRITE_EMULATION and map_path.exists() and history_path.exists():
            try:
                with np.load(history_path, allow_pickle=False) as h:
                    mean_history = np.asarray(h['mean_l1_error_percent'], dtype=np.float64)
                    scale_history = np.asarray(h['scale_l1_error_percent'], dtype=np.float64)
                    required_metadata = {'harmonic_iter', 'nside', 'lmax', 'n_iterations', 'source_tag', 'random_seed', 'scale_names', 'coarse_arcmin', 'target_power_path', 'target_power_mtime_ns', 'l1_target_paths', 'l1_target_mtime_ns'}
                    missing_metadata = required_metadata - set(h.files)
                    compatible = len(missing_metadata) == 0
                    incompatibility_reasons = []
                    if missing_metadata:
                        incompatibility_reasons.append('missing metadata: ' + ', '.join(sorted(missing_metadata)))
                    if compatible:
                        checks = [(int(h['harmonic_iter']) == HARMONIC_ITER, 'HARMONIC_ITER'), (int(h['nside']) == EMULATION_NSIDE, 'EMULATION_NSIDE'), (int(h['lmax']) == EMULATION_LMAX, 'EMULATION_LMAX'), (int(h['n_iterations']) == N_ITERATIONS, 'N_ITERATIONS'), (str(h['source_tag']) == source_token(), 'source tag'), (int(h['random_seed']) == expected_random_seed, 'random seed'), (np.array_equal(np.asarray(h['scale_names'], dtype=str), np.asarray(scale_names, dtype=str)), 'L1 scale names'), (np.isclose(float(h['coarse_arcmin']), float(EMULATION_COARSE_ARCMIN)), 'coarse scale'), (str(h['target_power_path']) == str(TARGET_POWER_NPZ), 'target power-spectrum path'), (int(h['target_power_mtime_ns']) == int(target_power_mtime_ns), 'target power-spectrum file version'), (np.array_equal(np.asarray(h['l1_target_paths'], dtype=str), np.asarray([str(path) for path in l1_target_paths], dtype=str)), 'L1 target paths'), (np.array_equal(np.asarray(h['l1_target_mtime_ns'], dtype=np.int64), l1_target_mtime_ns), 'L1 target file versions')]
                        for ok, label in checks:
                            if not ok:
                                compatible = False
                                incompatibility_reasons.append(label)
                    if compatible:
                        print(f'[Realisation {realisation_index:03d}] existing compatible result loaded.', flush=True)
                        return {'realisation_index': realisation_index, 'map_path': str(map_path), 'history_path': str(history_path), 'mean_history': mean_history, 'scale_history': scale_history, 'skipped': True}
                    print(f'[Realisation {realisation_index:03d}] existing result is incompatible: ' + '; '.join(incompatibility_reasons), flush=True)
            except Exception as exc:
                print(f'[Realisation {realisation_index:03d}] existing result ignored: {exc}', flush=True)
        target_cls = np.zeros(EMULATION_LMAX + 1, dtype=np.float64)
        target_cls[required_ells] = target_all_cls[realisation_index, pos]
        target_cls[:2] = 0.0
        np.random.seed(expected_random_seed)
        print(f'[Realisation {realisation_index:03d}] generating Gaussian start map...', flush=True)
        solution = hp.synfast(target_cls, nside=EMULATION_NSIDE, lmax=EMULATION_LMAX, new=True, pol=False).astype(np.float64, copy=False)
        solution -= np.mean(solution)
        mean_history = []
        scale_history = []
        for iteration in range(N_ITERATIONS):
            t0 = time.perf_counter()
            components = decompose_map(solution, EMULATION_NSIDE, EMULATION_LMAX, BANDS, EMULATION_COARSE_ARCMIN)
            errors = []
            for name in scale_names:
                components[name], err = adjust_map_l1(components[name], TARGET_L1[name])
                errors.append(float(err) * 100.0)
            solution_l1 = components['fine'].copy()
            for name in scale_names:
                solution_l1 += components[name]
            solution_cls = adjust_cls(solution, EMULATION_NSIDE, EMULATION_LMAX, target_cls)
            solution_next = 0.5 * (solution_l1 + solution_cls)
            solution_next -= np.mean(solution_next)
            errors = np.asarray(errors, dtype=np.float64)
            mean_error = float(np.mean(errors))
            scale_history.append(errors)
            mean_history.append(mean_error)
            detail = ', '.join((f'{name}={err:.6f}%' for name, err in zip(scale_names, errors)))
            print(f'[Realisation {realisation_index:03d}] Iteration {iteration + 1:02d}/{N_ITERATIONS:02d} | Mean L1 error={mean_error:.6f}% | {detail} | time={time.perf_counter() - t0:.2f}s', flush=True)
            del solution, components, solution_l1, solution_cls
            solution = solution_next
            gc.collect()
        mean_history = np.asarray(mean_history, dtype=np.float64)
        scale_history = np.asarray(scale_history, dtype=np.float64)
        np.save(map_path, np.asarray(solution, dtype=np.float32))
        np.savez_compressed(history_path, realisation_index=np.int64(realisation_index), iterations=np.arange(1, N_ITERATIONS + 1, dtype=np.int64), mean_l1_error_percent=mean_history, scale_l1_error_percent=scale_history, scale_names=np.asarray(scale_names, dtype=str), harmonic_iter=np.int64(HARMONIC_ITER), nside=np.int64(EMULATION_NSIDE), lmax=np.int64(EMULATION_LMAX), n_iterations=np.int64(N_ITERATIONS), source_tag=np.asarray(source_token()), random_seed=np.int64(expected_random_seed), coarse_arcmin=np.float64(EMULATION_COARSE_ARCMIN), target_power_path=np.asarray(str(TARGET_POWER_NPZ)), target_power_mtime_ns=target_power_mtime_ns, l1_target_paths=np.asarray([str(path) for path in l1_target_paths], dtype=str), l1_target_mtime_ns=l1_target_mtime_ns, map_path=np.asarray(str(map_path)))
        print(f'[Realisation {realisation_index:03d}] SAVED: {map_path}', flush=True)
        del solution
        gc.collect()
        return {'realisation_index': realisation_index, 'map_path': str(map_path), 'history_path': str(history_path), 'mean_history': mean_history, 'scale_history': scale_history, 'skipped': False}
    if N_JOBS == 1:
        results = [run_one_emulation(i) for i in REALISATION_INDICES]
    else:
        results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10)((delayed(run_one_emulation)(i) for i in REALISATION_INDICES))
    results = sorted(results, key=lambda x: x['realisation_index'])
    print('=' * 96)
    print('EMULATION COMPLETED')
    print('=' * 96)
    return results



DEFAULT_DIAGNOSTIC_WAVELET_BANDS = (
    (15.0, 30.0),
    (18.0, 36.0),
    (20.0, 40.0),
    (25.0, 50.0),
)


def compute_theoretical_wavelet_l1_scales(
    config: WelcomeConfig,
    bands=DEFAULT_DIAGNOSTIC_WAVELET_BANDS,
    save: bool = True,
):
    """Compute theoretical wavelet L1 curves at arbitrary band-pass scales.

    This helper is intended for theory-versus-emulation diagnostics. The scales
    do not need to be identical to the wavelet constraints used during map
    reconstruction.
    """
    _apply_config(config)
    _configure_source()
    context = ThreeScaleTheoryContext.build()

    products = {}
    flat_save = {}
    for theta1, theta2 in bands:
        theta1 = float(theta1)
        theta2 = float(theta2)
        if not (0.0 < theta1 < theta2):
            raise ValueError(f"Invalid wavelet band: ({theta1}, {theta2}).")
        label = f"{theta1:g}-{theta2:g}"
        product = wavelet_theory_product(context, theta1, theta2, f"diagnostic_{label}")
        products[label] = product
        token = f"wavelet_{theta1:g}_{theta2:g}".replace(".", "p")
        flat_save[f"{token}_kappa"] = np.asarray(product["kappa"], dtype=np.float64)
        flat_save[f"{token}_pdf"] = np.asarray(product["pdf"], dtype=np.float64)
        flat_save[f"{token}_l1"] = np.asarray(product["l1"], dtype=np.float64)

    save_path = None
    if save:
        save_path = L1_DIR / f"diagnostic_wavelet_l1_{source_token()}.npz"
        np.savez_compressed(save_path, **flat_save)
        print("[SAVED]", save_path)

    return {
        "bands": tuple((float(a), float(b)) for a, b in bands),
        "products": products,
        "path": save_path,
    }


def run_welcome(config: WelcomeConfig | None = None):
    """Run the complete theory-driven WELCOME pipeline and save all products."""
    if config is None:
        config = WelcomeConfig()
    _apply_config(config)
    _configure_source()
    power_product = compute_theoretical_power_spectra()
    wavelet_product = compute_theoretical_wavelet_l1()
    emulation_results = generate_emulated_maps(config)
    return {
        "config": config,
        "power_spectrum_path": TARGET_POWER_NPZ,
        "wavelet_l1_path": THEORY_ALL_SCALE_NPZ,
        "map_directory": EMULATION_OUTPUT_DIR,
        "power_product": power_product,
        "wavelet_product": wavelet_product,
        "emulation_results": emulation_results,
    }
