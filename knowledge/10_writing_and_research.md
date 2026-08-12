# Writing & Published Interpretability Research

I write up my interpretability work publicly. The blog at itsme-aniketghosh.github.io/blog is an index that points readers to wherever each piece actually lives. The research itself goes on LessWrong (lesswrong.com/users/aniket-ghosh), because that's where people who'll actually argue with it are. Two research posts are live as of August 2026.

The reason this matters: I don't only do interpretability for coursework. I run my own experiments end to end and publish them somewhere they'll get read and picked apart.

## Post 1 — "Function Vectors as a Model-Diffing Tool: 17 Heads Repair a Bad Fine-Tune" (LessWrong, Aug 2026)
My strongest result, and the one most relevant to safety work. Standard function-vector work patches activations across contexts inside a single model. I extended it to patch across two models: copying activations from a clean base model into a LoRA that had been deliberately trained to give bad medical advice, which is an emergent-misalignment setup.

A guarded greedy search over 1,008 patchable units turns up 17 attention heads in layers 12–19 that repair the behavior. Transplanting only those heads moved the misaligned model from 3 to 27 safe responses out of 36 on held-out questions, against a clean-model ceiling of 30. ARC-Easy accuracy stayed flat at 100%, so this isn't lobotomizing the model to make it harmless. It beat the alternatives on the same task too: crosscoder diffing managed 1 out of 36, few-shot prompting 23, steering 10.

The part I find most interesting is an asymmetry. The intervention cures misalignment but can't induce it. Run in reverse, trying to inject the bad behavior into the clean model, it scores −0.2 logits, while a matched random direction scores +19.9 on the same test. So the method is worse than random at causing harm. (That +19.9 is the random baseline. It is never a result of the method.) The effect only runs in the repair direction, which says something about how the misalignment is stored in the first place.

Tested on Qwen2.5-7B, replicated on Llama-3.1-8B. Link: lesswrong.com/posts/6iPEuEnguEtmtaqyJ/function-vectors-as-a-model-diffing-tool-17-heads-repair-a

What I like about it is that it takes a technique people already use for explanation and turns it into something operational — find the bad fine-tune, undo it — and that the odd asymmetric result is one I checked rather than smoothed over.

## Post 2 — "Linear Probes Tell You Where Quantization Will Hurt" (LessWrong, Jul 2026)
Interpretability with an immediate engineering payoff: use probing to decide where you're allowed to compress a model.

I trained per-layer linear probes for token-level tasks (NER, POS, chunking) to find where that signal actually sits in the residual stream, then allocated precision by layer according to the map — bits only where the signal lives. That held 99–100% accuracy at a 5-bit average. Uniform quantization on the same budget collapsed to 16–41% on unseen datasets.

Link: lesswrong.com/posts/oJJyYDgPD95jEfvQx/linear-probes-tell-you-where-quantization-will-hurt

Knowing where a capability lives tells you what you can touch and what you can't. That's the kind of interpretability I want to do: findings that change a decision, not just describe one.

## How to talk about this
Asked whether I have published research or public artifacts in AI safety: yes, two LessWrong writeups in 2026, plus a first-author Springer paper and a co-authored IEEE paper with a patent filed.

Asked what I'd bring to an interpretability or evals team: I've run the whole loop myself. Hypothesis, experiment, baselines I actually tried to beat, replication on a second model, and a public writeup that has to survive other people reading it.

Still on the list: a public writeup of the SCD resume-screening bias audit, which is a strong evals result that currently only lives in the project repo.
