# Stochastic Calculus & Itô's Lemma — Code

Companion code for the blog post  
**[An Introduction to Stochastic Calculus and Itô's Lemma](https://tuanle618.github.io/blog/2026/stochastic-calculus-ito-lemma/)**

## Files

| File | Description |
|------|-------------|
| `stochastic_calculus.py` | Self-contained Python script with all simulations |
| `stochastic_calculus.ipynb` | Jupyter notebook version |

## Sections

1. **Brownian Motion** — sample paths of pure Wiener process  
2. **Stochastic Integrals (Itô Isometry)** — empirical verification: $g(s)=1$ and $g(s)=s$  
3. **Ornstein-Uhlenbeck Process** — mean-reverting SDE via Euler-Maruyama  
4. **Brownian Motion on a Circle** — Itô's Lemma in action; exact (angular) vs. Euler-Maruyama  
5. **Circle Parameterization** — static diagram of the unit-circle setup  

## Usage

```bash
pip install numpy matplotlib
python stochastic_calculus.py          # runs all simulations and saves figures
```

Figures are saved to `./figures/` (created automatically).
