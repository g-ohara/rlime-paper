# rlime-paper

Paper of **R-LIME**, a novel method for interpreting ML models[^1].
R-LIME explains behavior of black-box classifiers such as deep neural networks or ensemble models.
It linearly approximizes a decision boundary of the black-box classifier in a local rectangular region, and maximizes the region as long as the approximation accuracy is higher than a given threshold.
Then, it provides contribution of each feature to the prediction and rule that restricted the approximation region.

[^1]: Ohara, G., Kimura, K., Kudo, M. (2025). R-LIME: Rectangular Constraints and Optimization for Local Interpretable Model-agnostic Explanation Methods. In: Antonacopoulos, A., Chaudhuri, S., Chellappa, R., Liu, CL., Bhattacharya, S., Pal, U. (eds) Pattern Recognition. ICPR 2024. Lecture Notes in Computer Science, vol 15315. Springer, Cham. https://doi.org/10.1007/978-3-031-78354-8_6

### Related repositories

- [g-ohara/rlime](https://github.com/g-ohara/rlime)
- [g-ohara/rlime-examples](https://github.com/g-ohara/rlime-examples)
- [g-ohara/rlime-poster](https://github.com/g-ohara/rlime-poster)
- [g-ohara/rlime-slides](https://github.com/g-ohara/rlime-slides)
- [g-ohara/rlime-ga](https://github.com/g-ohara/rlime-ga)

## How to compile

### Prerequisites

* $\TeX{}$ compiler
* latexmk
* (optional) ChkTeX
* (optional) latexindent

### How to compile locally

1. Create your repository from tex-template
1. Edit `main.tex`
1. Run: `latexmk main.tex`

### How to compile on GitHub

- Tag your commit.
- Push the tag.
- Wait a few minutes.

## Versioning Rule

```
+------- public submission (eg. conference, presentation)
| +----- private submission (eg. review request for professor)
| | +--- large changes
0.0.0
```

