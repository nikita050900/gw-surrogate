# gw-surrogate

Neural network surrogate model for gravitational waveforms.

The idea: gravitational waveform models are expensive to evaluate.
For parameter estimation pipelines that need millions of waveform
evaluations, a trained neural network that approximates the waveform
can be orders of magnitude faster. This is the same concept behind
surrogate/emulator models used in power systems simulation, climate
modeling, and other domains where the forward model is a computational
bottleneck.

This project builds a simple surrogate for inspiral-phase gravitational
waveforms as a learning exercise. It follows the framework described in
Khan & Green (2021) but uses a stripped-down waveform model and a small
fully-connected network trainable on a laptop CPU.

## Status

Work in progress.

## Setup

conda env create -f environment.yml
conda activate gw-surrogate

## Structure

- `src/generate_data.py` — generate training waveforms
- `src/model.py` — PyTorch network architecture
- `src/train.py` — training loop
- `src/evaluate.py` — validation, mismatch metric, timing comparison

## References

- Khan & Green (2021), "Gravitational-wave surrogate models powered by
  artificial neural networks." Phys. Rev. D 103, 064015.
