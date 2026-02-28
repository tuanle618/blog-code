# Stochastic Lorenz System — Code

Companion code for the blog post  
**[The Stochastic Lorenz System: Chaos Meets Randomness](https://tuanle618.github.io/blog/2026/stochastic-lorenz-generative-images/)**

## Files

| File | Description |
|------|-------------|
| `lorenz_system.py` | Full Python module (`LorenzSystem`, `LorenzPlotter`, `NoiseType`) |
| `lorenz_system.ipynb` | Jupyter notebook with example usage |

## Quick start

```python
from lorenz_system import LorenzSystem, LorenzPlotter, NoiseType

lorenz  = LorenzSystem()
plotter = LorenzPlotter(gradient_name="pastel")

# Deterministic
result = lorenz.solve(t_span=(0, 45), n_points=9000)
plotter.plot_3d(result, title="Deterministic Lorenz Attractor")

# Additive noise
result = lorenz.solve(noise_type=NoiseType.ADDITIVE, D=3.0, t_span=(0, 45), dt=0.005)
plotter.plot_3d(result, title="Additive Noise (D=3.0)")

# Multiplicative noise
result = lorenz.solve(noise_type=NoiseType.MULTIPLICATIVE, D=0.02, t_span=(0, 45), dt=0.005)

# Ornstein-Uhlenbeck colored noise
result = lorenz.solve(noise_type=NoiseType.ORNSTEIN_UHLENBECK, D=5.0, correlation_time=5.0,
                      t_span=(0, 45), dt=0.005)
plotter.plot_2d_projections(result)
```

## Requirements

```bash
pip install numpy matplotlib
pip install plotly          # optional — for interactive_3d plots
```

## Noise types

| `NoiseType` | SDE formulation |
|-------------|-----------------|
| `NONE` | Deterministic Lorenz |
| `ADDITIVE` | $dX = f(X)dt + \sqrt{2D}dW_t$ |
| `MULTIPLICATIVE` | $dX = f(X)dt + X\sqrt{2D}dW_t$ |
| `ORNSTEIN_UHLENBECK` | Lorenz driven by OU-colored noise with correlation time $\tau$ |
