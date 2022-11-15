# AdvLatGAN: Adversarial Latent Generative Adversarial Networks
Official implementation of **NeurIPS 2022 Spotlight** paper "Improving Generative Adversarial Networks via Adversarial Learning in Latent Space".

![sampling_shift](./figures/sampling_shift.jpg)

**A Brief Introduction:** This work integrates adversarial techniques on latent space with GAN to improve the generation performance. The generation pipeline suffers from the "too continuous" issue when it tries to match up with the real data distribution, which is supported on disjoint manifolds. Adopting adversarial techniques in latent space, we impose an extra (implicit) transform function on the raw Gaussian sampling in GANs to achieve generation performance gain. Introducing targeted sampling transform in GAN training alleviates training challenges and empowers more robust network training pipelines, while the sampling transform in inference (generation) time directly improve the generation quality. 

## Code Organization

The implementation consists of three parts:

- AdvLatGAN-qua: GAN training algorithm for better quality
- AdvLatGAN-div: GAN training algorithm for more diverse generation
- AdvLatGAN-z: post-training latent space sampling improvement

Experiments on -qua and -div are based on different implementations and we separate the code into folders `AdvLatGAN-qua&-z` and `AdvLatGAN-div`.  `AdvLatGAN-qua&-z`  also includes the code of -z.

## Run

To run the code, please refer to the `README.md` in the subdirectories.

## Acknowledgements

This repository is built upon [pytorch-gan-collections](https://github.com/w86763777/pytorch-gan-collections), [pytorch-gan-metrics](https://github.com/w86763777/pytorch-gan-metrics) and [MSGAN](https://github.com/HelenMao/MSGAN).
