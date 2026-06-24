# AI Safety Direction

## Why safety, and what I mean by it
As models get more capable, the bottleneck stops being "can it do the task" and becomes "can we trust what it's doing and why." That's the problem I want to work on. Concretely I'm drawn to **mechanistic interpretability** (understanding model internals causally, not just correlationally) and **evaluations** (measuring real capabilities, failure modes, and bias rather than vibes). I'm not interested in safety as a slogan — I'm interested in the empirical, technical version of it.

## What I've actually done here (not just interest)
- **Mechanistic interpretability on Llama-3-8B (SCD project):** built an LDA + gradient-attribution + causal-patching pipeline and produced a real result — that probing accuracy and causal relevance are distinct (perfect probe, 0% causal flip rate). This is the methodological core of interpretability work.
- **Bias / evals:** audited an LLM resume-screener with counterfactual name-swaps and quantified systematic name-based bias (95.8% flip rate on identical-content resumes). That's an evaluation of a real safety-relevant failure mode.
- **Teaching the ethics-technical bridge:** I co-designed the AIDE curriculum covering algorithmic fairness (COMPAS, FairLearn), differential privacy, and LLM interpretability (LogitLens) for philosophy PhD students — so I can also explain this work to non-technical stakeholders.

## Credentials and trajectory
- **Selected for BlueDot Impact — Technical AI Safety** (cohort, 2026). I treat the program as an audition: the goal is a citable public interpretability/evals artifact, a facilitator reference, and network intros — not just a line on the resume.
- **Targets I'm working toward:** ARENA (self-study now), Apart Research hackathons, SPAR, and ultimately MATS / Anthropic Fellows-style research-engineer programs, with a full-time safety/evals/interpretability role as the goal by graduation (May 2027).
- **Plan:** publish the SCD bias-audit as a public writeup (LessWrong / Alignment Forum + portfolio), keep producing small reproducible interpretability results, and build toward a research-engineer role on a safety or evals team.

## How this fits an employer
If you're a safety, evals, or interpretability team: I already speak the language and have shipped a real (if small) result, I can code the experiments, and I can communicate findings clearly. If you're a product/research team that cares about reliability: I bring the same rigor — counterfactual testing, causal checks, honest measurement — to whatever you ship.
