# An Uncertain Future: Forecasting from Static Images using Variational Autoencodersutoencoders.md

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

### The network Architecture

Consists of 3 sub-networks:

* Image tower
* Encoder tower
* Decoder tower

### Quantitative Results - Negative Log Likelihood

This model is evaluated in the context of generative models (like language models): Whether the method estimates a distribution where the ground truth is highly probable.

The method of this paper outperforms both baselines.

### Quantitative Results - Min Euclidean Distance

Regular Euclidean distance is not appropriate for this multimodal output. For that reason "Euclidean distance of the closest sample to the fround truth and averrage over all the testing images" is used.

The thing I don't undrestant in this paper is that they evaluate this metric over upto 800 samples, why so many?

### Qualitative Results

They sample 800 times and use KMeans (K = 10) to come up with the possible futures.

To me this seems a bit excessive. Can't things like "Determinantal Point Process" help with coming up with the 10 most diverse samples more efficiently?

They also show that interpolating between latent variable values interpolates between the motion prediction which is nice.

## References

[1] 
