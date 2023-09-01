# Transformer
**Paper:** [Attention is all you need](https://arxiv.org/abs/1706.03762).

## Key points
- Introduced the transformer architecture.
- Scaled dot-product attention. $$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$
- Multi-head attention. $$Multihead(Q,K,V)=Concat(head_1, \dots, head_h)W^0$$, where $$head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$$
- Feed-forward sublayer. $$FFN(x)=max(0, xW_1 + b_1)W_2 + b_2$$
- Sinusoidal positional encoding. $$PE_{(pos,2i)} = sin(\frac{pos}{10000^{\frac{2i}{dmodel}}})$$ $$PE_{(pos,2i+1)} = cos(\frac{pos}{10000^{\frac{2i}{dmodel}}})$$
- Shared weights between encoder and decoder embeddings and output linear layer.
- Usage of label smoothing of value $\epsilon_{ls}=0.1$ during the cross-entropy computation.

## BLEU scores
The BLEU scores are based on the test set of the WMT14 dataset. 
| Model | en-de | en-fr |
| - | - | - |
| Base | 27.3 | 38.1 |
| Big | 28.4 | 41.8 |

## Citation
```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```