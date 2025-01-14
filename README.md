# Spatial-Transformer-Networks-with-Keras

This repository provides a Colab Notebook that shows how to use [Spatial Transformer Networks (STN)](https://arxiv.org/abs/1506.02025) inside CNNs build in Keras.
The Colab Notebook has been obtained from forking [Spatial-Transformer-Networks-with-Keras](https://github.com/sayakpaul/Spatial-Transformer-Networks-with-Keras).

I have used utility functions mostly from [this repository](https://github.com/kevinzakka/spatial-transformer-network) to demonstrate an end-to-end example.
As such please install stn from pypi:
```
pip3 install stn
```

STNs allow a (vision) network to learn the optimal spatial transformations for maximizing its performance. In other words, we can expect when STNs are incorporated inside a network, it would learn how much to rotate or crop (or any affine transformations) the given input images so as to make itself more invariant to these changes.

Here's a demonstration:

https://user-images.githubusercontent.com/22957388/115120399-e8084b80-9fca-11eb-97e1-c72228c3edc4.mov

Notice how the STN module is able to figure out transformations for the dataset that may be helpful to boost its performance. Here are the original images for reference:

<div align="center">
<img src="https://i.ibb.co/1bQys44/image.png"></img>
</div>

This repository has also been updated to use a STN as a Keras Layer.
