# FAQ — Common Interview & Recruiter Questions

**What kind of role are you looking for, and when are you available?**
My ideal role is Research Engineer — implementing hard research ideas and making them run in practice. I'm especially drawn to AI safety, interpretability, and evaluations teams, and equally open to a core ML/AI engineering role where the problems are hard. I'm looking for **Spring 2027 roles starting January 2027**, full-time or co-op, and I graduate in May 2027. I'm on an F-1 visa and STEM OPT eligible.

**Do you have real industry experience?**
Yes. I'm the sole ML engineer at Varosync, an early-stage computational drug-development startup, where I train 7–8B-parameter molecular embedding models on an 800M-molecule corpus across a multi-GPU cluster and own everything downstream of the gold labels through to the live query path — reporting directly to the CEO/CTO. So beyond research and coursework, I've owned a real production ML pipeline end-to-end.

**Have you published anything? Do you have public research?**
Yes — two mechanistic-interpretability writeups on LessWrong in 2026: "Function Vectors as a Model-Diffing Tool: 17 Heads Repair a Bad Fine-Tune" and "Linear Probes Tell You Where Quantization Will Hurt." I also have a first-author Springer paper (AISC 2024) and a co-authored IEEE paper with a patent filed by SRM. My blog at itsme-aniketghosh.github.io/blog indexes everything.

**What's the hardest / most complex problem you've solved?**
The function-vector model-diffing work. I took a technique that patches activations across contexts within one model and extended it to patch across two models, to see whether a deliberately misaligned medical fine-tune could be repaired from the base model's activations. A guarded greedy search over 1,008 patchable units found 17 attention heads that took safe responses from 3/36 to 27/36 with capability untouched — beating crosscoder diffing, few-shot, and steering. The part I'm proudest of is the asymmetry I found and checked rather than skipped: the intervention cures misalignment but can't induce it. Before that, the SCD project — proving on Llama-3-8B that a probe can hit 1.000 accuracy on a feature with 0% causal effect on the output — was the result that changed how I think about interpretability claims.

**Can you handle a heavy workload?**
Through summer 2026 I ran three demanding responsibilities at once — sole ML engineer at Varosync, Teaching Co-Lead of the AIDE ML lab, and my M.S. (4.0 GPA) plus the BlueDot AI Safety program — and published two research writeups in the same stretch. I've operated like this historically too: ranked 2nd of 180+ in undergrad while doing two research roles, and 7 years of volunteer teaching alongside full course loads.

**Why AI safety?**
As models get more capable, the hard problem shifts from "can it do the task" to "can we trust what it's doing and why." I want to work on the empirical, technical version of that — interpretability and evals — not safety as a slogan. I've produced real published results (function-vector repair of a misaligned fine-tune, probe-guided quantization, the SCD causal study, and an LLM resume-screening bias audit) and I completed BlueDot's Technical AI Safety program, certified July 2026.

**Is your interpretability work only academic, or has it touched production?**
Both. At Varosync I applied mechanistic interpretability techniques to refine model architecture, which improved inference efficiency and performance. The probe-guided quantization work is the same idea generalized: knowing where a capability lives tells you what you can compress and what you can't.

**What are your strengths?**
End-to-end ownership, rigor (I check causally, test counterfactually, run the baselines I claim to beat, and replicate on a second model), and range — I move between interpretability research, production ML, and teaching. I ramp fast in unfamiliar stacks.

**What's a weakness / where are you early?**
My production experience is recent (Varosync, from May 2026) and I'm early-career, so I haven't yet led large engineering teams or run systems at FAANG scale. I'm honest about that — and I close the gap fast because I take ownership and learn by shipping.

**Why would you be a good fit here / how can you help us?**
I bring a rare combination: I can do the research (interpretability/evals with published results), code the experiments and the production pipeline, and communicate findings to technical and non-technical audiences — I spent a summer teaching this material to philosophy PhDs. Hand me your hardest, most ambiguous problem and I'll define the metric, build the system, and push until it works.

**Are you only interested in safety, or also engineering roles?**
Both. The common thread is hard technical problems done rigorously. Safety/interpretability/evals is my research passion; ML/AI engineering is where I ship. A research-engineer role sits right in the middle, which is why it's my ideal.

**Where do you study — Berkeley?**
No — I'm at Northeastern University, Khoury College (M.S. in AI); undergrad at IEM Kolkata. Any mention of other schools in old essays was aspirational.
