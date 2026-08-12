# Writing & Published Interpretability Research

I write up my interpretability work publicly. My blog is at **itsme-aniketghosh.github.io/blog** — it's an index that routes readers to wherever each piece actually lives. The research itself is published on **LessWrong** (profile: lesswrong.com/users/aniket-ghosh), which is where the alignment/interpretability discussion happens. Two research posts are live as of August 2026.

This matters as a signal: I don't just do interpretability in coursework, I run my own experiments end to end and publish results where researchers in the field will actually read and critique them.

## Post 1 — "Function Vectors as a Model-Diffing Tool: 17 Heads Repair a Bad Fine-Tune" (LessWrong, Aug 2026)
This is my strongest research result, and it's directly safety-relevant. Standard function-vector work patches activations **across contexts within one model**; I extended it to patch **across two models** — copying activations from a clean base model into a LoRA fine-tune that had been deliberately trained to give bad medical advice (an emergent-misalignment setup).

- A **guarded greedy search over 1,008 patchable units** finds **17 attention heads in layers 12–19** that repair the misaligned behavior. Transplanting just those heads took the misaligned model from **3 to 27 of 36 safe responses** on held-out questions, against a clean-model ceiling of 30.
- **Capability was preserved**: ARC-Easy accuracy held flat at 100%. So this isn't lobotomizing the model — it's a targeted repair.
- It **beat the alternatives** on the same task: crosscoder diffing (1/36), few-shot prompting (23/36), and steering (10/36).
- I established a **repair/injection asymmetry**: the intervention cures misalignment but **cannot induce it**. Run in reverse — trying to *inject* the bad behaviour into the clean model — it scores **−0.2 logits**, while a **matched random direction scores +19.9** on that same injection test. So at causing harm the intervention is *worse than a random direction*; the effect only runs in the repair direction. (Note: +19.9 is the random baseline, not a result of the method.) That asymmetry is the interesting part — it says something about how the misalignment is actually stored.
- Tested on **Qwen2.5-7B** and **replicated on Llama-3.1-8B**.
- Link: lesswrong.com/posts/6iPEuEnguEtmtaqyJ/function-vectors-as-a-model-diffing-tool-17-heads-repair-a

Why I'm proud of it: it takes an existing interpretability technique and turns it into something with a concrete safety use — localizing and undoing a bad fine-tune — and it has a clean negative/asymmetric result that I checked rather than glossed over.

## Post 2 — "Linear Probes Tell You Where Quantization Will Hurt" (LessWrong, Jul 2026)
An interpretability result with an immediate engineering payoff: use probing to decide **where** you're allowed to compress a model.

- I trained **per-layer linear probes** for token-level tasks (**NER, POS, chunking**) to localize where that signal actually lives in the residual stream.
- I then **allocated precision by layer** according to the probe map — keeping bits only in the layers that carry the signal.
- Result: **99–100% accuracy at a 5-bit average**, where **uniform quantization collapsed to 16–41%** on unseen datasets.
- Link: lesswrong.com/posts/oJJyYDgPD95jEfvQx/linear-probes-tell-you-where-quantization-will-hurt

Why it matters: it's a case where interpretability isn't just explanatory — knowing *where* a capability lives tells you what you can and can't touch. That's the kind of interpretability I want to do: findings that change a decision.

## How to talk about this
- If someone asks whether I have published research or public artifacts in AI safety: **yes — two LessWrong writeups in 2026**, plus a first-author Springer paper and a co-authored IEEE paper with a patent filed. See also the teaching/publications notes.
- If someone asks what I'd bring to an interpretability or evals team: I've run the full loop myself — hypothesis, experiment, baselines to beat, replication on a second model, and a public writeup that survives scrutiny.
- Still on my list to publish: a public writeup of the **SCD resume-screening bias audit**, which is a strong evals result that currently only lives in the project repo.
