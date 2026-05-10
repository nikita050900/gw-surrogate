"""
Generate training data: TaylorF2 inspiral gravitational waveforms.

The TaylorF2 approximant is an analytic frequency-domain waveform for
the inspiral phase of a compact binary coalescence, computed using the
stationary phase approximation to post-Newtonian (PN) theory.

We use it here because:
  1. Fully analytic -- no external GW software needed (just numpy)
  2. Standard inspiral approximation used in LIGO/Virgo analyses
  3. Physically meaningful waveforms parameterized by component masses

The surrogate model will learn to map (m1, m2) -> (log_amplitude, phase),
replacing the analytic computation with a single neural network forward pass.

Reference for TaylorF2 phase coefficients:
  Buonanno, Iyer, Ochsner, Pan & Sathyaprakash (2009), PRD 80, 084043
  (see also Cutler & Flanagan 1994, Poisson & Will 1995)
"""

import numpy as np
import h5py
from pathlib import Path
from time import time


# ---------- physical constants ----------
G_SI = 6.67430e-11       # m^3 kg^-1 s^-2
C_SI = 2.99792458e8      # m s^-1
MSUN_SI = 1.98892e30     # kg
MPC_SI = 3.08568e22      # m


def compute_taylorf2(m1_msun, m2_msun, f_array, d_mpc=1.0):
    """
    Compute the TaylorF2 frequency-domain inspiral waveform at 1.5PN order.

    This is the leading-order + two PN correction terms for the phase,
    and the Newtonian-order amplitude (f^{-7/6} power law).

    Parameters
    ----------
    m1_msun : float
        Primary mass in solar masses (m1 >= m2 by convention).
    m2_msun : float
        Secondary mass in solar masses.
    f_array : 1-d array
        Frequency grid in Hz.
    d_mpc : float
        Luminosity distance in Mpc (default 1.0; for the surrogate we
        only care about the waveform *shape*, not the overall scaling
        with distance, but we keep this physical so the amplitude has
        correct units of strain/Hz).

    Returns
    -------
    amplitude : array, same shape as f_array
        |h~(f)|  (one-sided amplitude spectral density)
    phase : array, same shape as f_array
        Psi(f) in radians
    """
    # --- derived mass quantities ---
    m1 = m1_msun * MSUN_SI
    m2 = m2_msun * MSUN_SI
    M_total = m1 + m2
    eta = (m1 * m2) / M_total**2          # symmetric mass ratio (0 < eta <= 0.25)
    M_chirp = M_total * eta**(3.0 / 5.0)  # chirp mass

    # total mass in seconds  (geometric units: G M / c^3)
    M_sec = G_SI * M_total / C_SI**3

    # chirp mass in seconds
    Mc_sec = G_SI * M_chirp / C_SI**3

    # --- PN expansion parameter ---
    # v = (pi M f)^{1/3},  the characteristic orbital velocity / c
    v = (np.pi * M_sec * f_array) ** (1.0 / 3.0)

    # --- phase: Psi(f) at 1.5PN ---
    # We set t_c = 0, phi_c = 0  (all waveforms aligned to same
    # coalescence time and reference phase).
    #
    # Psi(f) = -pi/4 + (3 / 128 eta) v^{-5}
    #          * [ 1
    #              + (3715/756 + 55 eta/9) v^2      <-- 1PN
    #              - 16 pi v^3                       <-- 1.5PN
    #            ]
    psi_prefactor = 3.0 / (128.0 * eta)
    pn_sum = (
        1.0
        + (3715.0 / 756.0 + 55.0 * eta / 9.0) * v**2   # 1PN
        - 16.0 * np.pi * v**3                             # 1.5PN
    )
    phase = -np.pi / 4.0 + psi_prefactor * v**(-5) * pn_sum

    # --- amplitude: leading (Newtonian) order ---
    # |h~(f)| = sqrt(5 pi / 24) * Mc^{5/6} / (pi^{2/3} D_L) * f^{-7/6}
    d_si = d_mpc * MPC_SI
    amplitude = (
        np.sqrt(5.0 * np.pi / 24.0)
        * Mc_sec ** (5.0 / 6.0)
        / (np.pi ** (2.0 / 3.0) * d_si)
        * f_array ** (-7.0 / 6.0)
    )

    return amplitude, phase


def generate_dataset(
    n_samples,
    f_min=20.0,
    f_max=300.0,
    n_freq=200,
    m1_range=(5.0, 20.0),
    m2_range=(5.0, 20.0),
    seed=42,
):
    """
    Generate a dataset of TaylorF2 waveforms with random (m1, m2).

    Sampling: uniform in the triangle  m2_min <= m2 <= m1,
    m1_min <= m1 <= m1_max.  This is the simplest unbiased choice.
    Latin hypercube sampling would give slightly better coverage
    but is unnecessary at 10k points in 2D.

    Returns a dict ready to be saved to HDF5.
    """
    rng = np.random.default_rng(seed)
    f_array = np.linspace(f_min, f_max, n_freq)

    # sample m1 uniformly, then m2 uniformly in [m2_min, m1]
    m1_samples = rng.uniform(m1_range[0], m1_range[1], size=n_samples)
    m2_samples = rng.uniform(m2_range[0], m1_samples)

    amplitudes = np.zeros((n_samples, n_freq))
    phases = np.zeros((n_samples, n_freq))

    for i in range(n_samples):
        amp, phi = compute_taylorf2(m1_samples[i], m2_samples[i], f_array)
        amplitudes[i] = amp
        phases[i] = phi

    # log-amplitude: the raw amplitude spans orders of magnitude
    # across the frequency band, so log10 brings it to a scale where
    # standardization (zero mean, unit variance) works well.
    log_amplitudes = np.log10(amplitudes)

    return {
        "params": np.column_stack([m1_samples, m2_samples]),  # shape (N, 2)
        "log_amplitude": log_amplitudes,                       # shape (N, n_freq)
        "phase": phases,                                       # shape (N, n_freq)
        "f_array": f_array,                                    # shape (n_freq,)
    }


def save_dataset(data, filepath):
    """Save dataset dict to HDF5."""
    with h5py.File(filepath, "w") as f:
        for key, val in data.items():
            f.create_dataset(key, data=val)
    n = data["params"].shape[0]
    print(f"  saved {n} waveforms -> {filepath}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    t0 = time()

    print("generating training set (10 000 waveforms) ...")
    train = generate_dataset(10_000, seed=42)
    save_dataset(train, out_dir / "train.h5")

    print("generating validation set (2 000 waveforms) ...")
    val = generate_dataset(2_000, seed=123)
    save_dataset(val, out_dir / "val.h5")

    print("generating test set (1 000 waveforms) ...")
    test = generate_dataset(1_000, seed=456)
    save_dataset(test, out_dir / "test.h5")

    elapsed = time() - t0
    print(f"\ndone in {elapsed:.1f}s")

    # quick sanity check: print parameter ranges and data shapes
    print(f"\n--- sanity check (training set) ---")
    print(f"  params shape:        {train['params'].shape}")
    print(f"  log_amplitude shape: {train['log_amplitude'].shape}")
    print(f"  phase shape:         {train['phase'].shape}")
    print(f"  m1 range: [{train['params'][:,0].min():.2f}, {train['params'][:,0].max():.2f}] Msun")
    print(f"  m2 range: [{train['params'][:,1].min():.2f}, {train['params'][:,1].max():.2f}] Msun")
    print(f"  freq range: [{train['f_array'][0]:.1f}, {train['f_array'][-1]:.1f}] Hz")
    print(f"  log_amp range: [{train['log_amplitude'].min():.2f}, {train['log_amplitude'].max():.2f}]")
    print(f"  phase range:   [{train['phase'].min():.1f}, {train['phase'].max():.1f}] rad")
