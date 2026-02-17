# Individual project notes

## Structure

### Feature extractors

cnn_feature_extractor.py: CNN backbone feature extractor

zernike_feature_extractor.py: Feature extraction based on Zernike Moments, which assists in acquiring rotation-invariant features

Feature extractors can be swapped out with DINO feature extractor depending on what works best, but I will try to implement them for a useful comparison.

### Deep Cross-Scale PatchMatch

TODO

## Paper link

https://arxiv.org/pdf/2404.17310

## Other useful resources

## CNN feature extractor:

Our backbone architecture comprises five convolution blocks, each consisting of a convolution layer, a BatchNorm layer, and a
ReLU layer. At the end of the backbone, there is a resizing
layer to rescale the feature maps to the same dimensions of
H × W ×c, where c represents the number of feature channels,
empirically set to 32

## Zernike feature extractor:

Maximal order is set to 5, yielding a 12-dimensional feature map. We compoute this for $I_u, I_o$ and $I_d$.
$\textbf{F}_{ZM}(p, q) = \frac{p + 1}{\pi}\sum_{(x, y) \in \Omega_{xy}} I(x, y)K^*_{p,q}(\rho_{xy}, \theta_{xy})$

Where:
$\Omega_{xy}$ is a set of pixels centered by (x, y)
$\rho_{xy} = \sqrt{x^2 + y^2}$
$K^*_{p,q}(\rho_{xy}, \theta_{xy})$ is a filter, computed using:
$K_{p,q}(\rho, \theta) = R_{p, q}(\rho)\exp(jm\theta)$
And $R_{p, q}(\rho) = \sum_{s=0}^{(1 - |p|)/2} \frac{(-1)^s[(1-s)!]\rho^{1-2s}}{s!(\frac{1+|p|}{2}-s)!(\frac{1-|p|}{2}+s)!}$ (looks scary but is just numerical)

This can be calculated as a convolution of the filters over the image pixels.

UPDATE: Looks like these might be incorrect:
https://mathworld.wolfram.com/ZernikePolynomial.html?utm_source=chatgpt.com

_The Zernike feature extractor is implemented based on the formula for R shown above, and other formulas come from the paper. Some LLM code generation was used for assistance in the polar grid computation as well as the forward pass implementation._
_In addition, an LLM was used to help optimize the implementation to not exceed available VRAM on an RTX 4070 TI SUPER_
