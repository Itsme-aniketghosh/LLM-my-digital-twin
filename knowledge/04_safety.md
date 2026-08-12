# AI Safety Direction

## Why safety, and what I mean by it
As models get more capable, the hard question stops being "can it do the task" and becomes "can we tell what it's doing and why." That's the problem I want. Concretely: mechanistic interpretability, meaning understanding internals causally rather than correlationally, and evaluations, meaning measuring real capabilities and failure modes instead of going on vibes. Safety as an engineering problem, not safety as a position.

## What I've actually done here
Repaired a misaligned fine-tune with function-vector model diffing, published on LessWrong in August 2026. Transplanting 17 attention heads from the clean base model into a LoRA trained to give bad medical advice moved safe responses from 3/36 to 27/36, against a clean ceiling of 30, with ARC-Easy capability flat at 100%. It beat crosscoder diffing, few-shot, and steering. And it cures misalignment without being able to induce it, which I checked and then replicated on a second model. Emergent misalignment with a mitigation attached.

Probe-guided quantization, published July 2026. Per-layer linear probes locate where token-level capability lives, then precision gets allocated to match: 99–100% accuracy at a 5-bit average, against 16–41% for uniform quantization.

The SCD project on Llama-3-8B, where an LDA plus gradient-attribution plus causal-patching pipeline showed that probing accuracy and causal relevance come apart. Perfect probe, zero causal effect. That's a caution about a method the field leans on heavily.

An LLM resume-screener audit using counterfactual name swaps, which found a 95.8% flip rate on identical-content resumes. A safety-relevant failure mode in a system people actually deploy today.

A grokking reproduction where I reverse-engineered a 1-layer transformer computing a discrete Fourier transform for modular addition. Behavior tells you nothing about the algorithm.

And the AIDE ML lab, where I taught algorithmic fairness, differential privacy, sparse autoencoders, the logit lens, and linear probing with nnsight and NDIF to philosophy PhD students. Which means I can also explain this work to the people who have to reason about it without a CS background.

## Credentials and trajectory
BlueDot Impact's Technical AI Safety program: completed, certified July 2026. I treated it as an audition and it produced what I wanted, which was public research rather than another line on a résumé.

The tooling I actually use day to day is nnsight, NDIF, PyTorch hooks, DAS and Boundless DAS and MDAS, causal patching, linear probes, crosscoders, and SAEs.

Next: ARENA, Apart Research hackathons, SPAR, and MATS or Anthropic Fellows-style research-engineer programs, aiming at a full-time safety, evals, or interpretability role. I can start January 2027, co-op or full-time, and I graduate in May.

## How this fits an employer
For a safety, evals, or interpretability team: I already speak the language, I've published real results, I can build the experiments, and I can write them up so they hold up.

For a product or research team that cares about reliability: same rigor, different subject. Counterfactual testing, causal checks, baselines I genuinely try to beat, honest measurement. I've already carried interpretability into production once, using it to refine model architecture at Varosync, so this isn't theoretical for me.
