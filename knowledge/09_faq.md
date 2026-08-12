# FAQ — Common Interview & Recruiter Questions

**What kind of role are you looking for, and when are you available?**
Research Engineer is the one I want: implementing hard research ideas and making them run in practice. AI safety, interpretability, and evaluations teams first, though a core ML or AI engineering role suits me just as well if the problems are hard. I'm looking at Spring 2027 roles starting January 2027, full-time or co-op, and I graduate in May 2027. F-1 visa, STEM OPT eligible.

**Do you have real industry experience?**
Yes. I'm the only ML engineer at Varosync, an early-stage computational drug-development startup. I train 7–8B-parameter molecular embedding models on an 800M-molecule corpus across a multi-GPU cluster and own everything downstream of the gold labels through to the live query path, reporting to the CEO and CTO. So this isn't research and coursework with nothing behind it.

**Have you published anything?**
Two mechanistic-interpretability writeups on LessWrong in 2026: "Function Vectors as a Model-Diffing Tool: 17 Heads Repair a Bad Fine-Tune" and "Linear Probes Tell You Where Quantization Will Hurt." Also a first-author Springer paper from AISC 2024 and a co-authored IEEE paper with a patent filed by SRM. My blog at itsme-aniketghosh.github.io/blog indexes all of it.

**What's the hardest problem you've solved?**
The function-vector model-diffing work. Function vectors normally patch activations across contexts inside one model, and I extended it to patch across two, to find out whether a deliberately misaligned medical fine-tune could be repaired using the clean base model's activations. A guarded greedy search over 1,008 patchable units found 17 attention heads that took safe responses from 3/36 to 27/36 with capability untouched, beating crosscoder diffing, few-shot, and steering.

The part I'm proudest of is the asymmetry I found and then checked instead of skipping: the intervention cures misalignment but can't induce it. Before that, the SCD result — proving on Llama-3-8B that a probe can hit 1.000 accuracy on a feature with zero causal effect on the output — is the one that changed how I read other people's interpretability claims.

**Can you handle a heavy workload?**
Through summer 2026 I ran three demanding things at once: sole ML engineer at Varosync, Teaching Co-Lead of the AIDE ML lab, and the M.S. at a 4.0 plus BlueDot's AI Safety program. I published two research writeups in the same stretch. It's not new either. I ranked 2nd of 180+ in undergrad while holding two research roles, and I taught for seven years at The Hope Foundation alongside full course loads.

**Why AI safety?**
Because as models get more capable the hard question shifts from "can it do the task" to "can we tell what it's doing and why." I want the empirical version of that problem, interpretability and evals, not safety as a stance. I have results to point at: function-vector repair of a misaligned fine-tune, probe-guided quantization, the SCD causal study, and an LLM resume-screening bias audit. I completed BlueDot's Technical AI Safety program and was certified in July 2026.

**Is your interpretability work only academic, or has it touched production?**
Both. At Varosync I used mechanistic interpretability to refine model architecture, which improved inference efficiency and performance. The probe-guided quantization work is the same idea generalized: knowing where a capability lives tells you what you can compress and what you can't.

**What are your strengths?**
End-to-end ownership. Rigor, meaning I check causally, test counterfactually, run the baselines I claim to beat, and replicate on a second model. And range: I move between interpretability research, production ML, and teaching, and I get productive in an unfamiliar stack fast.

**Where are you weakest?**
My production experience started in May 2026, so it's real but it's about a year deep. I haven't led a large engineering team or run systems at FAANG scale. I'd rather say that plainly than dress it up. What I'd point to instead is how fast I close that kind of gap once I own something.

**Why would you be a good fit here?**
The combination is the rare part. I can do the research, with published results in interpretability and evals. I can write the experiments and the production pipeline. And I can explain the findings to people who don't have a CS background, because I spent a summer teaching exactly this material to philosophy PhDs. Give me the hardest, most ambiguous thing on the list and I'll define the metric, build the system, and stay on it until it works.

**Are you only interested in safety, or also engineering roles?**
Both, and the common thread is hard technical problems done carefully. Safety, interpretability, and evals is the research I care about; ML and AI engineering is where I ship. Research Engineer sits in the middle of those, which is why it's the role I want.

**Where do you study — Berkeley?**
No. Northeastern University, Khoury College, for the M.S. in AI, and IEM Kolkata for undergrad. Any mention of other schools in old essays was aspirational.
