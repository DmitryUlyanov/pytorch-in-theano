## Pytorch inside Theano

<small>(and pytorch wrapper for [AIS](https://github.com/tonywu95/eval_gen))</small>

This repo shows a dirty hack, how to run Pytorch graphs inside any Theano graph. Moreover, both forward and backward passes are supported. So whenever you want to mix Pytorch and Theano you may use the wrapper from this repo.

In particular, I wanted to use [this code](https://github.com/tonywu95/eval_gen) to evaluate a generative model using [1]. Their code is written in Theano and my [AGE](https://arxiv.org/abs/1704.02304) model was trained using Pytorch.

[1] *On the Quantitative Analysis of Decoder-Based Generative Models*, Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov, Roger Grosse, ICLR 2016

## General usage
As easy as it could be.
```
from ops import pytorch_wrapper
f_theano = pytorch_wrapper(f_pytorch, dtype=dtype, debug=True)
```
And then use `f_theano` in your theano graphs. See `simple_example.py`.
## As AIS wrapper on MNIST dataset

0. Train your pytoch model on MNIST dataset.

1. Clone this repo with `--recursive` flag:
```
git clone --recursive https://github.com/DmitryUlyanov/pytorch_in_theano
```
2. See `age.py` for an example how I used it for AGE model on MNIST dataset. You will need to replace `NetWrapper(net)` in `generator(z)` with your network.    


If you want to compare your generative model, here are the two likelihood scores I've computed for MNIST with `z_dim=10`:

| method | score |
|--------|-------|
|  [AGE](https://arxiv.org/abs/1704.02304)   |  746  |
|  ALI   |  721  |

And the results from the paper: 705 for VAE and 328 for GAN.
### On other datasets
To be true, I now do not remember why I had to [modify sampler](https://github.com/DmitryUlyanov/eval_gen/commit/2347d967ef5554719cb6c4fa1a12f0a7b7903939) in `eval_gen` code. But probably it is because of shapes mismatch errors, that I struggled to figure out for a long time. So, please, before blindly running the code examine sampler file. For sure you need to put [here](https://github.com/DmitryUlyanov/eval_gen/blob/master/sampling/samplers_32.py#L109) the right value and probably change something in several other places.

## Misc

Tested with python 2, `theano=0.8.2.dev-901275534cbfe3fbbe290ce85d1abf8bb9a5b203`, `pytorch=0.2.0_4`.


If you find this code helpful for your research, please cite this repo:

```
@misc{Ulyanov2017_ais_wrapper,
  author = {Ulyanov, Dmitry},
  title = {Pytorch wrapper for AIS},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DmitryUlyanov/pytorch-in-theano}},
}
```
