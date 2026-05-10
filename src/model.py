"""
Neural network surrogate for gravitational waveforms.

Architecture: fully-connected MLP
  Input:   2  (normalised m1, m2)
  Hidden:  256 -> 256 -> 256 -> 256  (4 layers, ReLU activation)
  Output:  400 (200 log-amplitude + 200 phase values, normalised)

Why this architecture?
  - MLP because the input is a low-dimensional parameter vector, not an
    image or sequence.  No spatial/temporal structure to exploit.
  - 4 hidden layers: enough depth for composing nonlinear features of the
    PN waveform (which involves powers of mass combinations).  Shallower
    networks (2 layers) may underfit the phase; deeper ones aren't needed
    for a 2D input space.
  - 256 neurons per layer: modest width, sufficient for 400 outputs.
    Total parameters ~ 230k -- trains in minutes on CPU.
  - ReLU: fast, standard, no vanishing-gradient issues for 4 layers.
  - No dropout / batch-norm: training data is exact (no measurement noise)
    and we have 10k samples for 2 inputs, so overfitting is unlikely.
    Add regularisation only if val loss diverges from train loss.

This is the same class of architecture used in Khan & Green (2021)
for their SEOBNRv4 surrogate, scaled down for our simpler waveform model.
"""

import torch
import torch.nn as nn


class WaveformSurrogate(nn.Module):
    """
    Fully-connected network:  (m1, m2) -> (log_amp, phase) at fixed freqs.
    """

    def __init__(self, input_dim=2, output_dim=400, hidden_dim=256, n_hidden=4):
        super().__init__()

        layers = []

        # input -> first hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # hidden -> hidden
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # last hidden -> output (no activation -- regression output)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor, shape (batch, 2)
            Normalised (m1, m2).

        Returns
        -------
        y : tensor, shape (batch, 400)
            Normalised (log_amplitude, phase) concatenated.
        """
        return self.network(x)

    def count_parameters(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# quick sanity check when run directly
if __name__ == "__main__":
    model = WaveformSurrogate()
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # test forward pass with dummy input
    x = torch.randn(8, 2)      # batch of 8, 2 inputs
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
