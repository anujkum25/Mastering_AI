This repository has following purpose:
- read all about LLMs and implemennt the same from scratch
- implement llm papers
- implement diffusion model papers



24-Week Plan (LLMs + Diffusion, Open-Source Focus)
🔹 Phase 1 (Weeks 1–4): Foundations

Goal: Get comfortable with PyTorch, re-implement basic models, prep for Transformers.

Week 1

PyTorch crash course: tensors, autograd, optimizers

Implement: Linear regression + logistic regression from scratch

Mini-project: Train a 2-layer MLP on MNIST

Week 2

Study: Backpropagation, initialization, overfitting/regularization

Implement: CNN (LeNet) on CIFAR-10 from scratch

Paper skim: LeNet-5 (1998)

Week 3

Study: Sequence models — RNN, LSTM, GRU

Implement: Char-level RNN for text generation

Paper skim: Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)

Week 4

Study: Attention mechanism basics

Implement: Scaled dot-product attention manually

Paper: Attention is All You Need (Vaswani, 2017)

🔹 Phase 2 (Weeks 5–8): Transformers & LLM Basics

Goal: Build transformers & small LLMs from scratch.

Week 5

Study: Transformer encoder-decoder architecture

Implement: Transformer encoder (mini version) for classification

Reproduce: The Annotated Transformer (Harvard NLP blog)

Week 6

Study: Causal language modeling (GPT-1 style)

Implement: GPT mini (causal decoder-only transformer)

Train on Tiny Shakespeare dataset

Week 7

Study: Tokenization (BPE, WordPiece)

Implement: Your own BPE tokenizer

Plug tokenizer into your GPT-mini

Week 8

Paper skim: GPT-2 (2019)

Implement: Scale GPT-mini → GPT-small (~10–20M params)

Add: Gradient clipping, learning rate warmup

🔹 Phase 3 (Weeks 9–12): Scaling & Advanced LLM Training

Goal: Learn scaling laws, stability tricks, and Hugging Face integration.

Week 9

Study: LayerNorm, weight initialization, residual connections

Implement: Add LayerNorm, residuals to GPT

Reproduce training curves

Week 10

Study: Scaling laws (Kaplan et al., 2020)

Implement: Mixed precision training (torch.cuda.amp)

Train GPT-small on WikiText-2

Week 11

Hugging Face focus:

Read Transformers source code for GPT2Model

Implement a custom model and push to Hugging Face Hub

Learn Trainer API

Week 12

Project: Fine-tune GPT-2 on your own dataset with Hugging Face

Implement: LoRA fine-tuning from scratch

Paper skim: LoRA (Hu et al., 2021)

🔹 Phase 4 (Weeks 13–16): Diffusion Foundations

Goal: Build diffusion models from scratch on small images.

Week 13

Study: Generative models overview (VAEs, GANs, Diffusion)

Implement: VAE on MNIST (prep for diffusion)

Paper skim: Auto-Encoding Variational Bayes (Kingma, 2013)

Week 14

Study: DDPM basics (forward process, reverse process)

Paper: Denoising Diffusion Probabilistic Models (Ho et al., 2020)

Implement: DDPM from scratch (MNIST)

Week 15

Implement: UNet backbone for DDPM

Add: Timestep embeddings (sinusoidal)

Train DDPM on CIFAR-10

Week 16

Paper: Improved DDPM (Nichol & Dhariwal, 2021)

Implement: Improved variance schedules, noise prediction

Compare FID scores

🔹 Phase 5 (Weeks 17–20): Advanced Diffusion & Text Conditioning

Goal: Implement text-conditioned diffusion (Stable Diffusion basics).

Week 17

Study: Classifier-free guidance (Ho, 2021)

Implement: CFG in your DDPM

Train class-conditional CIFAR-10 diffusion

Week 18

Paper: Latent Diffusion Models (Rombach et al., 2022)

Study: Autoencoders for latent space

Implement: Latent diffusion on MNIST latent space

Week 19

Implement: Text conditioning with embeddings (use pretrained BERT embeddings for conditioning)

Generate simple text-to-image samples

Week 20

Hugging Face Diffusers focus:

Read UNet2DConditionModel source code

Implement a minimal version

Contribute: bug fix, small feature, or doc improvement

🔹 Phase 6 (Weeks 21–24): Cutting Edge + Open Source Contributions

Goal: Reproduce new papers & actively contribute to Hugging Face repos.

Week 21

Paper skim: Consistency Models (Song et al., 2023)

Implement: Consistency model for CIFAR-10

Compare with DDPM

Week 22

Paper skim: DiT – Diffusion Transformers (Peebles & Xie, 2023)

Implement: Vision transformer backbone for diffusion

Experiment on CIFAR-10

Week 23

LLM + Diffusion crossover:

Implement text embeddings → condition diffusion pipeline

Build toy text-to-image diffusion with your GPT-mini embeddings

Week 24

Contribute to Hugging Face

Open a PR: bug fix, new config, example script, or new model

Engage on GitHub issues/discussions

Write a blog summarizing your re-implementations

📌 Daily / Weekly Rhythm

Mon–Wed → Paper reading + coding from scratch

Thu–Fri → Train & debug experiments

Sat → Read Hugging Face source code / contribute

Sun → Summarize learnings in notes/blog