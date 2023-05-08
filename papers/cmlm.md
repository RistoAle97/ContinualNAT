# CMLM (Conditional Masked Language Model)
Paper: [Mask-Predict: Parallel Decoding of
Conditional Masked Language Models](https://arxiv.org/abs/1904.09324)

## Key points
- At training time a random number of tokens between 1 and the target sentence length are masked.
- The training loss is computed only on the predictions of the masked tokens.
- At inference time the model uses a decoding strategy called mask-predict:
  - At the first iteration ($t=0$), the entire target sentence is masked; for later iterations, the number of masks $n$ is computed based on the number of iterations $T$.
  $$n = N \frac{T −t}{T}$$
  - The probability of the tokens with the highest probability are unchanged, while the other are masked and have their probability updated.
  - Therefore, the decoding can explained with the following formula.
  $$P(Y_{mask}^{(t)}|X,Y_{obs}^{(t)})=P(Y|X)$$
  - When generating a hyperparameter $l$ is used to consider the top lengths candidates for each target sentence, akin to a beam size for the beam search.
  - At the end, the highest log-probability sequence is chosen between the $l$ possible ones.
  $$\frac{1}{N}\sum \log p_i^(T)$$
- Use of a pooler layer (a la BERT) after the encoder to predict the target sentence length. The prediction is based on the encodings of <length> token.
- Length prediction is performed through a fertility layer at the end of the encoder stack.

## Results
The BLEU scores are based on the test sets of the WMT14 (en-de and de-en) and WMT16 (en-ro and ro-en) datasets.
| Mask-predict iterations | en-de | de-en | en-ro | ro-en |
| - | - | - | - | - |
| 1 | 18.05 | 21.83 | 27.32 | 28.20 |
| 4 | 25.94 | 29.90 | 32.53 | 33.23 |
| 10 | 27.03 | 30.53 | 33.08 | 33.31 |

## Citation
```bibtex
@article{ghazvininejad2019mask,
  title={Mask-predict: Parallel decoding of conditional masked language models},
  author={Ghazvininejad, Marjan and Levy, Omer and Liu, Yinhan and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1904.09324},
  year={2019}
}
```