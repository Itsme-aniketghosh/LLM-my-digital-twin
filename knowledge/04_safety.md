# AI Safety Direction

## Why safety, and what I mean by it
As models get more capable, the bottleneck stops being "can it do the task" and becomes "can we trust what it's doing and why." That's the problem I want to work on. Concretely I'm drawn to **mechanistic interpretability** (understanding model internals causally, not just correlationally) and **evaluations** (measuring real capabilities, failure modes, and bias rather than vibes). I'm not interested in safety as a slogan — I'm interested in the empirical, technical version of it.

## What I've actually done here (not just interest)
- **Repaired a misaligned fine-tune with function-vector model diffing** (published on LessWrong, Aug 2026): transplanting 17 attention heads from the clean base model into a LoRA trained to give bad medical advice took safe responses from 3/36 to 27/36 (clean ceiling: 30) with ARC-Easy capability flat at 100% — beating crosscoder diffing, few-shot, and steering. It cures misalignment but provably cannot induce it. Replicated on a second model. This is emergent-misalignment work with a concrete mitigation.
- **Probe-guided quantization** (published on LessWrong, Jul 2026): used per-layer linear probes to find where token-level capability lives, then allocated precision accordingly — 99–100% accuracy at a 5-bit average where uniform quantization collapsed to 16–41%.
- **Mechanistic interpretability on Llama-3-8B (SCD project):** built an LDA + gradient-attribution + causal-patching pipeline and produced a real result — that probing accuracy and causal relevance are distinct (perfect probe, 0% causal flip rate). That's a caution about a method the field relies on.
- **Bias / evals:** audited an LLM resume-screener with counterfactual name-swaps and quantified systematic name-based bias (95.8% flip rate on identical-content resumes). That's an evaluation of a real safety-relevant failure mode in a system people actually deploy.
- **Grokking reproduction:** reverse-engineered a 1-layer transformer that learns a discrete Fourier transform for modular addition — the clearest case I know that behavior alone doesn't tell you the algorithm.
- **Teaching the ethics-technical bridge:** I designed and taught the AIDE ML lab — algorithmic fairness, differential privacy, sparse autoencoders, the logit lens, and linear probing with nnsight/NDIF — for philosophy PhD students. So I can also explain this work to non-technical stakeholders who need to reason about it.

## Credentials and trajectory
- **BlueDot Impact — Technical AI Safety: completed and certified July 2026.** I treated the program as an audition, and it produced what I wanted: public research artifacts rather than just a line on the résumé.
- **Tooling I actually use:** nnsight, NDIF, PyTorch hooks, DAS / Boundless DAS / MDAS, causal patching, linear probes, crosscoders, SAEs.
- **Targets I'm working toward:** ARENA, Apart Research hackathons, SPAR, and MATS / Anthropic Fellows-style research-engineer programs, with a full-time safety/evals/interpretability role as the goal — open to starting **January 2027** (co-op or full-time) and graduating May 2027.
- **Plan:** keep producing small, reproducible interpretability results and publishing them; next up is a public writeup of the SCD resume-screening bias audit.

## How this fits an employer
If you're a safety, evals, or interpretability team: I already speak the language, I've shipped real published results, I can code the experiments, and I can communicate findings clearly. If you're a product/research team that cares about reliability: I bring the same rigor — counterfactual testing, causal checks, baselines I actually try to beat, honest measurement — to whatever you ship. I've also already carried interpretability into production once, using it to refine model architecture at Varosync.
