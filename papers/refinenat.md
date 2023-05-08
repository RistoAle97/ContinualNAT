   # FT-NAT (Fertility NAT)
Paper: [Deterministic Non-Autoregressive Neural Sequence Modeling
by Iterative Refinement](https://arxiv.org/abs/1802.06901)

## Key points
- First paper about iterative non-autoregressive NMT.
- During inference the target sentence is predicted based on the previous generated tokens.
- A second decoder stack is used to predict the target sentence length.
- Use of highway connections instead of residual ones.
- A higher number of decoding iteraions implies a better BLEU score at the expense of a worse speedup.

## Results
The BLEU scores are based on the test sets of the WMT14 (en-de and de-en) and WMT16 (en-ro and ro-en) datasets.
| Iterations | en-de | de-en | en-ro | ro-en |
| - | - | - | - | - |
| 1 | 13.91 | 16.77 | 24.45 | 25.73 |
| 2 | 16.95 | 20.39 | 27.10 | 28.15 |
| 5 | 20.26 | 23.86 | 28.86 | 29.72 |
| 10 | 21.61 | 25.48 | 29.32 | 30.19 |

## Citation
```bibtex
@article{lee2018deterministic,
  title={Deterministic non-autoregressive neural sequence modeling by iterative refinement},
  author={Lee, Jason and Mansimov, Elman and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1802.06901},
  year={2018}
}
```