# Individual project notes

## Structure

### Feature extractors

cnn_feature_extractor.py: CNN backbone feature extractor

zernike_feature_extractor.py: Feature extraction based on Zernike Moments, which assists in acquiring rotation-invariant features

Feature extractors can be swapped out with DINO feature extractor depending on what works best, but I will try to implement them for a useful comparison.

### Deep Cross-Scale PatchMatch

NB! It is okay that the candidates are sometimes not legit / go out of image.
We just evaluate the out-of bounds ones to 0 or something.
Remember that we add four individual 0-order candidates, 8 first-order and then another four from the random generation.
We also have the original offset - this means that we are likely to get something good anyway

Then, calculate the $\mathcal{l}_1$ norm between feature vectors.
Do this individually for each feature vector type!

NB! Test running this propagation and evaluation for different amount of iterations - original paper used 3-5 iterations

"To simplify, we define the resultant offset map from
CNN features as δ1, and the offset map from ZM features as
δ2. Figs. 5(b-c) illustrate two examples of the resulting offset
maps."

Note: $1 < \beta < 5$ are good candidates for the soft argmax. But be careful, we could get vanishing gradients

### DLF

Most likely interpretation of both diagram and paper - CNN features are used for DLF. Then the zernike offsets are just added as backup.

You can test yourself if whether removing them degrades performance.

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
