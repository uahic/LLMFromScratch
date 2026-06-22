# General

This is a from-scratch implementation of an LLM with the intent to gain experience in current optimization tricks and hyperparamter tweaking.
Unfortunately, I do only own 2x RTX 3090 (NVLink coupled) GPUs and not an entire datacenter but I might actually spend money deploying my models on rented hardware once I hit the most recent state-of-the-art wall. 
I'm using Stanfords CS336 (Language Modeling from Scratch) lecture and unit-test infrastructure as a guide-line but I will also put diffusion-transformer (not covered by this lecture) later into this or a seperate repository.

Current features:
- [x] vanilla dense all-to-all transformer
- [x] Rope positional embeddings
- [x] Multiprocessing BPE tokenizer (10k vocab trained on tinyStoriesV2(GPT4))
- [x] Numerical stable softmax + cross-entropy loss function
- [x] SwiGLU Linear Layer
- [x] Logging / Tensorboard
- [x] Checkpointing

Needs refinement:
- Logging / Tensorboard

Not implemented yet:
- [ ] Sharding (well, 'applying' it, not implementing this really from scratch)
- [ ] Linear Attention
- [ ] Mixture of Experts
- [ ] KV Cache for inference
- [ ] Fine-tuning training
- [ ] Tasks via Reinforcment-Learning (although some RL algorithms have been implemented in my other repos, I might copy them over, lets see)


Might land in another repo:
- [ ] Diffusion Transformer  

