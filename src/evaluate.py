"""
Evaluate the trained surrogate model against ground-truth waveforms.

Produces:
  1. Waveform comparison plots (surrogate vs analytic, amplitude + phase)
  2. Residual plots (prediction error vs frequency)
  3. Mismatch histogram across the test set
  4. Timing benchmark: surrogate vs analytic waveform generation

Usage:
    python src/evaluate.py
"""

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from time import time

from dataset import WaveformDataset
from model import WaveformSurrogate
from generate_data import compute_taylorf2


def load_trained_model(model_path, dataset):
    """Load best model checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = WaveformSurrogate(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded model from epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}")
    return model


def predict_all(model, dataset):
    """
    Run the surrogate on the full dataset and denormalize to physical units.

    Returns
    -------
    pred_log_amp : array (N, n_freq)
    pred_phase   : array (N, n_freq)
    true_log_amp : array (N, n_freq)
    true_phase   : array (N, n_freq)
    """
    with torch.no_grad():
        x = torch.from_numpy(dataset.params_norm)
        y_norm = model(x).numpy()

    pred_log_amp, pred_phase = dataset.denormalize_targets(y_norm)

    n = dataset.n_freq
    true_log_amp = dataset.targets[:, :n]
    true_phase = dataset.targets[:, n:]

    return pred_log_amp, pred_phase, true_log_amp, true_phase


def compute_mismatch(pred_log_amp, pred_phase, true_log_amp, true_phase, df):
    """
    Compute the gravitational-wave mismatch for each waveform pair.

    mismatch = 1 - |<h_true | h_pred>| / sqrt(<h_true|h_true> <h_pred|h_pred>)

    where <a|b> = sum_f  a*(f) b(f) df  (unweighted inner product).

    The absolute value maximises over a constant phase offset, which is
    the standard convention for assessing surrogate faithfulness.

    Parameters
    ----------
    pred_log_amp, pred_phase : arrays (N, n_freq)
    true_log_amp, true_phase : arrays (N, n_freq)
    df : float, frequency spacing in Hz

    Returns
    -------
    mismatches : array (N,)
    """
    # reconstruct complex waveforms from log-amplitude and phase
    amp_pred = 10.0 ** pred_log_amp
    amp_true = 10.0 ** true_log_amp

    h_pred = amp_pred * np.exp(1j * pred_phase)
    h_true = amp_true * np.exp(1j * true_phase)

    # inner products (sum over frequency axis)
    inner_tp = np.sum(np.conj(h_true) * h_pred, axis=1) * df
    inner_tt = np.sum(np.abs(h_true) ** 2, axis=1) * df
    inner_pp = np.sum(np.abs(h_pred) ** 2, axis=1) * df

    match = np.abs(inner_tp) / np.sqrt(inner_tt * inner_pp)
    mismatch = 1.0 - match

    return mismatch


def plot_waveform_comparisons(pred_log_amp, pred_phase, true_log_amp, true_phase,
                               params, freqs, out_dir, n_examples=4):
    """Plot surrogate vs ground truth for a few test waveforms."""
    rng = np.random.default_rng(0)
    indices = rng.choice(len(params), size=n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        m1, m2 = params[idx]

        # amplitude comparison
        ax = axes[row, 0]
        ax.plot(freqs, true_log_amp[idx], "k-", lw=1.5, label="analytic", alpha=0.8)
        ax.plot(freqs, pred_log_amp[idx], "r--", lw=1.5, label="surrogate", alpha=0.8)
        ax.set_ylabel("$\\log_{10} |\\tilde{h}|$")
        ax.set_title(f"$m_1={m1:.1f},\\; m_2={m2:.1f}\\;M_\\odot$ — amplitude")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # phase comparison
        ax = axes[row, 1]
        ax.plot(freqs, true_phase[idx], "k-", lw=1.5, label="analytic", alpha=0.8)
        ax.plot(freqs, pred_phase[idx], "r--", lw=1.5, label="surrogate", alpha=0.8)
        ax.set_ylabel("$\\Psi(f)$ [rad]")
        ax.set_title(f"$m_1={m1:.1f},\\; m_2={m2:.1f}\\;M_\\odot$ — phase")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    fig.savefig(out_dir / "waveform_comparison.png", dpi=150)
    print(f"saved waveform_comparison.png")
    plt.close()


def plot_residuals(pred_log_amp, pred_phase, true_log_amp, true_phase,
                   freqs, out_dir):
    """Plot residual (prediction - truth) statistics vs frequency."""
    amp_resid = pred_log_amp - true_log_amp   # in log10 units
    phase_resid = pred_phase - true_phase      # in radians

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # amplitude residuals: median + 5th/95th percentile band
    med = np.median(amp_resid, axis=0)
    p5 = np.percentile(amp_resid, 5, axis=0)
    p95 = np.percentile(amp_resid, 95, axis=0)
    ax1.plot(freqs, med, "b-", lw=1, label="median")
    ax1.fill_between(freqs, p5, p95, alpha=0.3, color="b", label="5th–95th pctl")
    ax1.axhline(0, color="k", ls="--", lw=0.5)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("$\\Delta \\log_{10} |\\tilde{h}|$")
    ax1.set_title("Amplitude residuals")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # phase residuals
    med = np.median(phase_resid, axis=0)
    p5 = np.percentile(phase_resid, 5, axis=0)
    p95 = np.percentile(phase_resid, 95, axis=0)
    ax2.plot(freqs, med, "r-", lw=1, label="median")
    ax2.fill_between(freqs, p5, p95, alpha=0.3, color="r", label="5th–95th pctl")
    ax2.axhline(0, color="k", ls="--", lw=0.5)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("$\\Delta \\Psi$ [rad]")
    ax2.set_title("Phase residuals")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    print(f"saved residuals.png")
    plt.close()


def plot_mismatch_histogram(mismatches, out_dir):
    """Histogram of mismatches across the test set."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(mismatches), bins=50, color="steelblue", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("$\\log_{10}$(mismatch)")
    ax.set_ylabel("count")
    ax.set_title(f"Mismatch distribution (test set, N={len(mismatches)})")

    # annotate summary stats
    med = np.median(mismatches)
    worst = np.max(mismatches)
    ax.axvline(np.log10(med), color="red", ls="--", lw=1.5,
               label=f"median = {med:.2e}")
    ax.axvline(np.log10(worst), color="orange", ls="--", lw=1.5,
               label=f"worst = {worst:.2e}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "mismatch_histogram.png", dpi=150)
    print(f"saved mismatch_histogram.png")
    plt.close()


def timing_benchmark(model, test_ds, n_repeats=100):
    """
    Compare inference time: surrogate vs analytic waveform generator.

    We time:
      1. The analytic TaylorF2 function for a single waveform
      2. The neural network forward pass for a single waveform
      3. The neural network forward pass for a batch of 1000 waveforms
    """
    freqs = test_ds.f_array
    m1, m2 = test_ds.params[0]

    # --- analytic timing ---
    # warm up
    compute_taylorf2(m1, m2, freqs)
    t0 = time()
    for _ in range(n_repeats):
        compute_taylorf2(m1, m2, freqs)
    t_analytic = (time() - t0) / n_repeats

    # --- surrogate single-waveform timing ---
    x_single = torch.from_numpy(test_ds.params_norm[:1])
    with torch.no_grad():
        model(x_single)  # warm up
    t0 = time()
    with torch.no_grad():
        for _ in range(n_repeats):
            model(x_single)
    t_surrogate_single = (time() - t0) / n_repeats

    # --- surrogate batch timing (1000 waveforms at once) ---
    x_batch = torch.from_numpy(test_ds.params_norm[:1000])
    with torch.no_grad():
        model(x_batch)  # warm up
    t0 = time()
    with torch.no_grad():
        for _ in range(n_repeats):
            model(x_batch)
    t_surrogate_batch = (time() - t0) / n_repeats
    t_surrogate_per_wf = t_surrogate_batch / 1000

    print(f"\n{'='*55}")
    print(f"  TIMING BENCHMARK  ({n_repeats} repeats, CPU)")
    print(f"{'='*55}")
    print(f"  analytic (1 waveform):    {t_analytic*1e6:8.1f} us")
    print(f"  surrogate (1 waveform):   {t_surrogate_single*1e6:8.1f} us")
    print(f"  surrogate (batch of 1000): {t_surrogate_batch*1e3:7.2f} ms "
          f"  ({t_surrogate_per_wf*1e6:.1f} us/waveform)")
    print(f"{'='*55}")
    print(f"  single-waveform speedup:  {t_analytic/t_surrogate_single:.1f}x")
    print(f"  batched speedup:          {t_analytic/t_surrogate_per_wf:.1f}x")
    print(f"{'='*55}")

    return {
        "analytic_us": t_analytic * 1e6,
        "surrogate_single_us": t_surrogate_single * 1e6,
        "surrogate_batched_us": t_surrogate_per_wf * 1e6,
        "speedup_single": t_analytic / t_surrogate_single,
        "speedup_batched": t_analytic / t_surrogate_per_wf,
    }


# ------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = Path("data")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # ── load data + model ───────────────────────────────────────
    print("loading test data ...")
    # load training set just to get norm stats
    train_ds = WaveformDataset(data_dir / "train.h5")
    test_ds = WaveformDataset(data_dir / "test.h5", norm_stats=train_ds.norm_stats)
    print(f"  {len(test_ds)} test waveforms")

    model = load_trained_model(data_dir / "best_model.pt", test_ds)

    # ── predict ─────────────────────────────────────────────────
    print("\nrunning surrogate on test set ...")
    pred_log_amp, pred_phase, true_log_amp, true_phase = predict_all(model, test_ds)

    # ── mismatch ────────────────────────────────────────────────
    df = test_ds.f_array[1] - test_ds.f_array[0]
    mismatches = compute_mismatch(pred_log_amp, pred_phase,
                                   true_log_amp, true_phase, df)

    print(f"\n--- mismatch summary (test set) ---")
    print(f"  median:  {np.median(mismatches):.2e}")
    print(f"  mean:    {np.mean(mismatches):.2e}")
    print(f"  worst:   {np.max(mismatches):.2e}")
    print(f"  best:    {np.min(mismatches):.2e}")
    pct_below_1e3 = 100 * np.mean(mismatches < 1e-3)
    pct_below_1e2 = 100 * np.mean(mismatches < 1e-2)
    print(f"  < 0.1%:  {pct_below_1e3:.1f}% of test set")
    print(f"  < 1%:    {pct_below_1e2:.1f}% of test set")

    # ── plots ───────────────────────────────────────────────────
    print("\ngenerating plots ...")
    plot_waveform_comparisons(pred_log_amp, pred_phase, true_log_amp, true_phase,
                               test_ds.params, test_ds.f_array, fig_dir)
    plot_residuals(pred_log_amp, pred_phase, true_log_amp, true_phase,
                   test_ds.f_array, fig_dir)
    plot_mismatch_histogram(mismatches, fig_dir)

    # ── timing ──────────────────────────────────────────────────
    timing = timing_benchmark(model, test_ds)

    print("\ndone.")
