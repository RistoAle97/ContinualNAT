# Joint Training
The following BLEU scores were obtained by the models using joint training (all the datasets are concatenated and shuffled).

---

## WMT14 test scores
Note that the scores were computed on the _newstest2014_ for all the translation directions except for $en\Leftrightarrow es$ for which the _newstest2013_ was used instead.

### Tokenizer 13a
|    Model    | $en\Rightarrow de$ | $de\Rightarrow en$ | $en\Rightarrow fr$ | $fr\Rightarrow en$ | $en\Rightarrow es$ | $es\Rightarrow en$ |
|:-----------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Transformer |       24.96        |       30.88        |       37.03        |       35.49        |        32.8        |       32.74        |
|    CMLM     |       18.42        |       26.01        |       29.94        |       31.05        |       27.23        |        28.6        |
|    GLAT     |       17.79        |        24.7        |       28.98        |       30.05        |       27.04        |       28.47        |

### Tokenizer intl
|    Model    | $en\Rightarrow de$ | $de\Rightarrow en$ | $en\Rightarrow fr$ | $fr\Rightarrow en$ | $en\Rightarrow es$ | $es\Rightarrow en$ |
|:-----------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Transformer |        25.6        |       31.31        |       39.99        |       36.19        |       32.87        |       33.18        |
|    CMLM     |       18.97        |       26.37        |       32.91        |        31.7        |       27.29        |       28.98        |
|    GLAT     |       18.31        |       24.98        |       31.85        |       30.66        |       27.04        |       28.47        |

---

## Flores200 devtest scores
### Tokenizer 13a
|    Model    | $en\Rightarrow de$ | $de\Rightarrow en$ | $en\Rightarrow fr$ | $fr\Rightarrow en$ | $en\Rightarrow es$ | $es\Rightarrow en$ |
|:-----------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Transformer |       33.64        |       37.21        |       43.84        |       38.74        |       24.05        |       24.53        |
|    CMLM     |        25.0        |       31.61        |       34.01        |       34.21        |       18.72        |       20.65        |
|    GLAT     |       23.27        |       29.84        |       33.32        |       33.15        |       18.62        |       20.05        |

### Tokenizer intl
|    Model    | $en\Rightarrow de$ | $de\Rightarrow en$ | $en\Rightarrow fr$ | $fr\Rightarrow en$ | $en\Rightarrow es$ | $es\Rightarrow en$ |
|:-----------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Transformer |       33.81        |       37.53        |        45.5        |       39.08        |       24.03        |       24.83        |
|    CMLM     |       25.08        |       31.91        |       35.63        |       34.44        |       18.67        |       20.83        |
|    GLAT     |        23.3        |       30.09        |       34.87        |       33.45        |       18.56        |       20.21        |
