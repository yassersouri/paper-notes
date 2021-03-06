# Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks

NIPS 2016 (Oral)| [1607.02586](https://arxiv.org/abs/1607.02586)

Xue T, Wu J, Bouman K, Freeman W (MIT)

## Notes

![Figure 1 of the paper](vd-fig1.png)

* Tries to generate the next frame (raw pixel values) from the current frame.
* Actually it models the difference between two frames (difference image).
* It uses the VAE framework to do so.
* It introduces a Cross Convolutinal Layer ()which is not anything special).
* It breaks down the input image into 32 regions (not exactly, because the image is tranformed into 32 feature maps not regions, but the author used this language in his oral presentation) where each region is moved independently with a convolutional operation. The kernels of those convolutions are estimated from the _z_ value of the encoder network.
* The feature maps are generated by the "image encoder".
* The image encoder gets as input a pyramid (4 levels) of the input image. And produces 4 set of 32 feature maps.
* The "cross convolution" part of the network does the convlution with movement kernels.
* The motion decoder regresses the "difference image" from the convluted feature maps.
* The motion encoder sees both the difference image and the first frame.
* Input size is 128x128.
* At test time the _z_ is not actually sampled from a gaussian but from "an empirical distribution of _z_ over all training samples as an approximation to the prior" (I assumed that is a parzen window)
* In the oral presentation ([video](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Visual-Dynamics-Probabilistic-Future-Frame-Synthesis-via-Cross-Convolutional-Networks)) author shows some qualitative results where interpolating in _z_ actually is meaningful. Suggesting that the latent space _z_ actually carries semantic. This part is not mentioned in the paper.

## Pros

* Implementing the "cross convolution" part of the network seems to be very easy in Lasagne ([arbitrary expressions as layer parameters](http://benanne.github.io/2015/11/10/arbitrary-expressions-as-params.html)) or Tensorflow.
* The paper is generally well written.
* One minor issue with the way the paper is written is in part 4.2 they must have first described the "image encoder" then "motion encoder and kernel decoder". Doing it the otherway makes the paper a bit confusing when you read it the first time.
* The design of the network is clever which is designed to model simple motion.


## Cons

* I don't understand why is it useful to generate future frames. Whatever we want to recognize from those future frames we can recognize directly. But it seems to be a cool thing to do and NIPS reviewers seems to have liked it.
* The _z_ dimension is very large i.e 3200! They show as part of their results that most of the z values are always zero.
* Motion encoder gets as input (during training) both the ground truth, *difference image* and also the *current frame*. It is not discussed in the paper what is the gain added by giving the *current frame* to the motion encoder. What happens if it is not included.
* This method only predicts the next frame, so long term motion is not predicted. To me long term motion is much more useful.
* They have only experimented with synthetic datasets or very simple objects moving in complete white background.
* The method is *designed* to be limited to very simple motion.
* Performance of the network is not discussed.


## Ideas

* What about MPEG? Couldn't we just predict the MPEG motion information given the reference frame? Shouldn't that be able to predict much more complicated scences?
