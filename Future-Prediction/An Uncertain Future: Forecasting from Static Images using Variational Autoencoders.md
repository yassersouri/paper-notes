# An Uncertain Future: Forecasting from Static Images using Variational Autoencodersutoencoders.md

ECCV 2016 | [1606.07873](http://arxiv.org/abs/1606.07873)

Walker J, Doersch G, Gupta A, Hebert M (CMU)

## High-level notes

* Predicting dense trajectory of pixels in a scene: What will move, where it will travel, how it will deform over the course of one second.
* Use conditional variational autoencoder to solve the problem.
* This method can produce multiple different predictions when the future is ambigous.
* The algorithm is trained on thousands of diverse, realistic videos.
* In this paper, they propose to develop a generative framework which, given a static input image, outputs the space of possible future dense optical flow predictions (which correspond to future actions).
* Sampling from this conditional generative model will result in multiple possible future events.
* One interesting line from the paper: "... it is unclear what is the output space of futures the algorithm should be capable of predicting." They think that optical flow or dense trajectories is a good thing to predict.


## Method


## References

[1] 
