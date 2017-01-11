# An Uncertain Future: Forecasting from Static Images using Variational Autoencodersutoencoders.md

ECCV 2016 | [1606.07873](http://arxiv.org/abs/1606.07873)

Walker J, Doersch G, Gupta A, Hebert M (CMU)

## High-level notes

* Predicting dense trajectory of pixels in a scene: What will move, where it will travel, how it will deform over the course of **one second**.
* Use **conditional variational autoencoder** to solve the problem.
* This method can produce multiple different predictions when the future is ambigous.
* The algorithm is trained on thousands of diverse, realistic videos.
* In this paper, they propose to develop a generative framework which, given a static input image, outputs the space of possible future dense optical flow predictions (which correspond to future actions).
* **Sampling** from this conditional generative model will result in multiple possible future events.
* One interesting line from the paper: "... it is unclear what is the output space of futures the algorithm should be capable of predicting." They think that optical flow or dense trajectories is a good thing to predict.
* The _conditional_ variational auto-encoder outputs a mapping from noise variables sampled from a gaussian to output trajectories at every pixel.
* They show that this method is able to do feature learning.

## Method
_X_: The image, _Y_: full set of trajectories

A regression network with gets as input an image + _z_ (which can account for ambiguity).
At test time _z_ is random Gaussian noise.
But training is not so traightforward.

### Training

The issue: given some ground-truth _Y_, we cannot directly measure the propability of the trajectory given an image _X_ with the current model. Which prevents using gradient descent on the likelihood. **Variational Autoencoders** are used to overcome this issue.

A new sub-network _Q_ is added. _Q_ is trained to "encode" Y into the latent space _z_ which then can be decoded by the regressor network. _l2_ norm reconstruction error can be used to train the network end-to-end.

**To prevent the model from simply copying**, (since it kind of seems Y during training): _Q_ produces a distribution over _z_ values which are sampled before decoding. Then information in this distribution is penelized by minimizing the KL-divergence between this distribution and _N(0, 1)_ (the trajectory agnostic distribution).

This to me feels like not a very smooth kind of optimization: first conver _Y_ to a set of samples, then compute KL-divergence, then 

### The model

At test time it is assumed as:

Y = \mu(X, z) + \epsilon

Where both z and \epsilon are white gaussian noise. \mu is implemented as a neural network.

## References

[1] 
