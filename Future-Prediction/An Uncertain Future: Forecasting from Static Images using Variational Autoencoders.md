# An Uncertain Future: Forecasting from Static Images using Variational Autoencoders

ECCV 2016 | [1606.07873](http://arxiv.org/abs/1606.07873)

Walker J, Doersch G, Gupta A, Hebert M (CMU)

## High-level notes

* Predicting dense trajectory of pixels in a scene: What will move, where it will travel, how it will deform over the course of **one second**.
* Use **conditional variational autoencoder** to solve the problem.
* This method can produce multiple different predictions when the future is ambigous.
* The algorithm is trained on thousands of diverse, realistic videos (not synthetic.)
* In this paper, they propose to develop a generative framework which, given a static input image, outputs the space of possible future dense optical flow predictions (which correspond to future actions).
* **Sampling** from this conditional generative model will result in multiple possible future events.
* One interesting line from the paper: "... it is unclear what is the output space of futures the algorithm should be capable of predicting." They think that optical flow or dense trajectories is a good thing to predict.
* The _conditional_ variational auto-encoder outputs a mapping from noise variables sampled from a gaussian to output trajectories at every pixel.
* They show that this method is able to do feature learning.

## Method
_X_: The image, _Y_: full set of trajectories

_Y_ is converted into the frequency-domain. (How exactly?)

A regression network which gets as input an image _X_ + _z_ (which can account for ambiguity).
At test time _z_ is random Gaussian noise.
But training is not so traightforward.

### Training

The issue: given some ground-truth _Y_, we cannot directly measure the propability of the trajectory given an image _X_ with the current model. Which prevents using gradient descent on the likelihood. **Variational Autoencoders** are used to overcome this issue.

A new sub-network _Q_ is added. _Q_ is trained to "encode" Y into the latent space _z_ which then can be decoded by the regressor network. _l2_ norm reconstruction error can be used to train the network end-to-end.

**To prevent the model from simply copying**, (since it kind of seems Y during training): _Q_ produces a distribution over _z_ values which are sampled before decoding. Then information in this distribution is penelized by minimizing the KL-divergence between this distribution and _N(0, I)_ (the trajectory agnostic distribution). (It is not clear for me why this is a good thing to do. Also not clear how to implement this efficiently)

### The model

At test time it is assumed as:

![Y = \mu(X, z) + \epsilon](https://chart.googleapis.com/chart?cht=tx&chl=Y%20%3D%20%5Cmu(X%2C%20z)%20%2B%20%5Cepsilon)

Where both z and ![e](https://chart.googleapis.com/chart?cht=tx&chl=\epsilon) are white gaussian noise. ![\mu](https://chart.googleapis.com/chart?cht=tx&chl=\mu) is implemented as a neural network.

_Q_ is set as, ![Q = \mathcal{N}(\mu'(X_i, Y_i), \delta'(X_i, Y_i))](https://chart.googleapis.com/chart?cht=tx&chl=Q%20%3D%20%5Cmathcal%7BN%7D(%5Cmu%27(X_i%2C%20Y_i)%2C%20%5Cdelta%27(X_i%2C%20Y_i))), where both ![\mu'](https://chart.googleapis.com/chart?cht=tx&chl=\mu%27) and ![\delta'](https://chart.googleapis.com/chart?cht=tx&chl=\delta%27) are implemented as neural networks (encoders).

Writing down ![E_{z \sim Q}[log P(Y_i|z, X_i)]](https://chart.googleapis.com/chart?cht=tx&chl=E_%7Bz%20%5Csim%20Q%7D%5Blog%20P(Y_i%7Cz%2C%20X_i)%5D) and using the Baye rule we get the standard variational equality.

![log P(Y_i|X_i) - \mathcal{KL}[Q(z|X_i, Y_i)  ||  P(z|X_i, Y_i)] =  E_{z \sim Q} [log P(Y_i|z, X_i)] - \mathcal{KL}[Q(z|X_i, Y_i) || P(z|X_i)]](https://chart.googleapis.com/chart?cht=tx&chl=log%20P(Y_i%7CX_i)%20-%20%5Cmathcal%7BKL%7D%5BQ(z%7CX_i%2C%20Y_i)%20%20%7C%7C%20%20P(z%7CX_i%2C%20Y_i)%5D%20%3D%20%20E_%7Bz%20%5Csim%20Q%7D%20%5Blog%20P(Y_i%7Cz%2C%20X_i)%5D%20-%20%5Cmathcal%7BKL%7D%5BQ(z%7CX_i%2C%20Y_i)%20%7C%7C%20P(z%7CX_i)%5D)

The gradient is computed with respect to only the right hand side of the equation. Doing gradient ascent, maximizes both sides.

If Q is high capacity enough, the KL-divergence can become zero, this means that the likelihood of Y is being optimized directly.

There is one assumption which is made here: _z_ is independent of X if Y in unknown.

**How to compute the gradient of _log P(Y|z, X)_?**

Values of z are sampled from Q. Using reparameterization trick we have:

![z_i = \mu'(X_i, Y_i) + \eta \circ \delta'(X_i, Y_i); \quad \eta \sim \mathcal{N}(0, 1)](https://chart.googleapis.com/chart?cht=tx&chl=z_i%20%3D%20%5Cmu%27(X_i%2C%20Y_i)%20%2B%20%5Ceta%20%5Ccirc%20%5Cdelta%27(X_i%2C%20Y_i)%3B%20%5Cquad%20%5Ceta%20%5Csim%20%5Cmathcal%7BN%7D(0%2C%201))

Now we can take the derivative of z with respect to ![\mu'](https://chart.googleapis.com/chart?cht=tx&chl=\mu%27) and ![\delta'](https://chart.googleapis.com/chart?cht=tx&chl=\delta%27).

### The network Architecture

Consists of 3 sub-networks:

* Image tower
	- Input is 320x240
	- AlexNet-like upto conv5 + 9 Conv (256, 3x3) layers.
	- Output shape: ?
* Encoder tower
	- Input: Frequency-domain trajectories, Downsampled spatially to be the same shape as the output of the "Image tower"
	- Outputs 2, 8 dimensional vectors for ![\mu'](https://chart.googleapis.com/chart?cht=tx&chl=\mu%27) and ![\delta'](https://chart.googleapis.com/chart?cht=tx&chl=\delta%27). (_z_ is assumed to be 8 dimensional) (Shape?)
	- AlexNet upto Conv5, + 2 Parallel Conv(8, 1x1) layers.
* Decoder tower
	- Input, replicated sampled _z_ values across spatial dimensions, multiplied with the output of the image tower with an offset. (?)
	- Conv(256, 3) -> Conv(256, 3) -> Conv(256, 3) -> Conv(256, 3) -> Conv(10, 3)
	- Output is 16x20 pixels. Which is split to normalized version of the trajectory (across time) plus the magnitude of the optical flow both horizontally and vertically.
	- There are 2 more coarse-to-fine branches for the decoder, which are trained only after the previous one has been trained and converged.

The output has three spatial resolutions: 1/16 (16 x 20 resolution), 1/8 and 1/4.

I have to say despite the authors effort to make it clear, details are still unclear for me to understand the details of the network.

**Loss**

![L(X, Y) = ||Y_{norm} - \hat{Y}_{norm} ||^2 + ||M_x - \hat{M}_x ||^2 + ||M_y - \hat{M}_y ||^2 + \mathcal{KL}[Q(z|X, Y) || \mathcal{N}(0, 1)]](https://chart.googleapis.com/chart?cht=tx&chl=L(X%2C%20Y)%20%3D%20%7C%7CY_%7Bnorm%7D%20-%20%5Chat%7BY%7D_%7Bnorm%7D%20%7C%7C%5E2%20%2B%20%7C%7CM_x%20-%20%5Chat%7BM%7D_x%20%7C%7C%5E2%20%2B%20%7C%7CM_y%20-%20%5Chat%7BM%7D_y%20%7C%7C%5E2%20%2B%20%5Cmathcal%7BKL%7D%5BQ(z%7CX%2C%20Y)%20%7C%7C%20%5Cmathcal%7BN%7D(0%2C%201)%5D)

### Implementation Details

Given videos, one-second (31-frame) clips are extracted, stabilized and trajectories (_Y_) are extracted using [1]. (Average trajectory over all scales is used.)

For each pixel in the first frame, they come up with x- and y-offset relative trajectory (60-dimensional vector for each pixel of a 1 second clip).

A discrete cosine transform is separately performed for x and y offsets and the first 5 components is choosen. Which results in 10-dimensional vector for each pixel of a 1 second clip.

Batch normalization is used after every convolutional layer (except the output layers where scale matters).

### Quantitative Results - Negative Log Likelihood

This model is evaluated in the context of generative models (like language models): Whether the method estimates a distribution where the ground truth is highly probable. They use Parzen window with Gaussian kernel to come up with the distribution from samples. They use the coarse resolution output (16 x 20) for this evaluation.

The method of this paper outperforms both baselines.

### Quantitative Results - Min Euclidean Distance

Regular Euclidean distance is not appropriate for this multimodal output. For that reason "Euclidean distance of the closest sample to the fround truth and averrage over all the testing images" is used.

The thing I don't undrestant in this paper is that they evaluate this metric over upto 800 samples, why so many?

### Qualitative Results

They sample 800 times and use KMeans (K = 10) to come up with the possible futures.

To me this seems a bit excessive. Can't things like "Determinantal Point Process" help with coming up with the 10 most diverse samples more efficiently?

They also show that interpolating between latent variable values interpolates between the motion prediction which is nice.

## References

[1] Wang, H., Schmid, C.: Action recognition with improved trajectories. ICCV 2013.
