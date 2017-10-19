# Pytorch inside Theano

This repo shows a dirty hack, how to run Pytorch graphs inside any Theano graph. Moreover, both forward and backward passes are supported. So whenever you want to mix Pytorch and Theano you may use the wrapper from this repo.

In particular, I wanted to use [this code](https://github.com/tonywu95/eval_gen) to evaluate a generative model using [1]. Their code is written in Theano and my model was trained using Pytroch.

[1] "On the Quantitative Analysis of Decoder-Based Generative Models", Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov, Roger Grosse, ICLR 2016

# General usage


# As AIS wrapper
