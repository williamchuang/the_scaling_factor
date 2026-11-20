# The Scaling Factor in Self-Attention

Paper by William Chuang (2023) and the notebooks behind its experiments.

## Question

Self-attention computes $\mathrm{softmax}(\beta\, QK^T)\,V$ with $\beta = 1/\sqrt{d_k}$. That value follows from four assumptions about the query and key matrices at initialization: unit variance, zero mean, and independence of their entries. Training moves $W_q$ and $W_k$ away from their initial distributions, so those assumptions stop holding once training starts. The paper asks whether $\beta$ should be fixed at $1/\sqrt{d_k}$ or tracked during training.

## Method

A **dynamic scaling factor**: measure the spread of $W_q$ and $W_k$ and update $\beta$ after each iteration or batch. A practical variant first tries $\beta \in \{10^{N}, 10^{-N}\}$ for $N = 0, 1, 2, \dots$ and then narrows $\beta$ by binary search.

## Results

Task: reverse a string of integers with a single multi-head attention model (fewer than 8,000 parameters). String length equals the model dimension $d_k$.

| β | Input range | String length | Training set | Test accuracy |
|---|---|---|---|---|
| $1/\sqrt{d_k}$ | 100 | 20 | 15,000 | 1.5% |
| 5 | 100 | 20 | 15,000 | 96% |
| $1/\sqrt{d_k}$ | 100 | 200 | 19,300 | 1.23% |
| 0.16 | 100 | 200 | 19,300 | 100% |
| $1/\sqrt{d_k}$ | 10 | 20 | 8,000 | 18.56% |
| 6.55 | 10 | 20 | 8,000 | 100% |
| $1/\sqrt{d_k}$ | 1000 | 20 | 35,000 | 5.52% |
| 6 | 1000 | 20 | 35,000 | 99.77% |

With the standard $\beta = 1/\sqrt{d_k}$, a stable test accuracy above 95% on the first task required 25,000 training samples instead of 15,000. Complete results are in Table 1 of the paper.

## Contents

| Path | Content |
|---|---|
| `paper/the_rescaling_factor.tex`, `.pdf` | Paper source and compiled paper |
| `paper/refs.bib` | Bibliography |
| `empirical_results/notebooks/` | Jupyter notebooks for models 123–137 |
| `empirical_results/*.pdf` | Exported runs of models 131–137 |

## Reproducing the experiments

```bash
pip install -r requirements.txt
jupyter notebook empirical_results/notebooks/
```

The notebooks use PyTorch and PyTorch Lightning. Model 136 uses a fixed scaling factor and model 137 a dynamic one.

## Citation

```bibtex
@misc{chuang2023scaling,
  author = {Chuang, William},
  title  = {The Scaling Factor in Self-Attention},
  year   = {2023},
  url    = {https://github.com/williamchuang/the_scaling_factor}
}
```

## License

MIT — see [LICENSE](LICENSE).
