# FT-NAT (Fertility NAT)
Paper: [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281)

## Key points
- First paper about non-autoregressive NMT.
- The target tokens are conditionally independent between each other during training.
$$\mathcal{L_NAT} = \sum_{t=1}^{T_Y} \log P(y_t|X;\theta)$$
instead of
$$\mathcal{L_AT}=\sum_{t=1}^{T_Y} \log P(y_t|y_{\< t},X;\theta)$$
- Sequence-level knowledge distillation to improve performances by tackling the multimodality problem.
- Length prediction is performed through a fertility layer at the end of the encoder stack.

## Results
The BLEU scores are based on the test sets of the WMT14 (en-de and de-en) and WMT16 (en-ro and ro-en) datasets.
| en-de | de-en | en-ro | ro-en |
| - | - | - | - |
| 17.35 | 20.62 | 26.22 | 27.83 |

## Citation
```bibtex
@article{gu2017non,
  title={Non-autoregressive neural machine translation},
  author={Gu, Jiatao and Bradbury, James and Xiong, Caiming and Li, Victor OK and Socher, Richard},
  journal={arXiv preprint arXiv:1711.02281},
  year={2017}
}
```