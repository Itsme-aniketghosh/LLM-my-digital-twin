# Projects

My two strongest research results are published on LessWrong — function-vector model diffing and probe-guided quantization — and are covered in the writing/research notes. The projects below are the rest of the portfolio.

## Sufficient Cause Disambiguation (SCD) — Mechanistic Interpretability
**CS 7180, PhD seminar with Prof. Byron Wallace · github.com/Itsme-aniketghosh/Sufficient-Cause-Disambiguation**
- **Proved probing accuracy and causal relevance are distinct properties** on Llama-3-8B: separator tokens reach 1.000 LDA accuracy across all 32 layers yet drive a **0% flip rate under causal patching**. In plain terms — a probe can read a feature perfectly while that feature has no causal effect on the model's output. That falsifies an assumption people lean on, and it matters a lot for interpretability claims.
- Built a full **mechanistic interpretability pipeline** (LDA + gradient attribution + causal activation patching), achieving **448× feature compression at 99.6% accuracy** and a **100% prediction-flip rate** at α=2.
- **Audited an LLM resume-screener for demographic bias** via counterfactual name-swap perturbations: identical-content resumes triggered a **95.8% prediction-flip rate** and accounted for **62% of the model's highest-uncertainty Fit decisions** — surfacing systematic name-based bias. This is a concrete evals/safety result, and a public writeup is still on my list.
- Why it's hard: causal patching is fiddly, and the negative result (probing ≠ causation) is the kind of thing that's easy to get wrong and important to get right.

## Grokking and Fourier Features in Neural Networks — Mechanistic Interpretability
**Independent research · reproduction of Nanda et al. (ICLR 2023) · github.com/Itsme-aniketghosh/grokking-fourier-circuits**
- Reproduced the grokking result and reverse-engineered the learned algorithm: a **1-layer transformer trained on modular addition internally implements a discrete Fourier transform** — an algorithm no one would guess from input/output behavior alone.
- Measured an **18:1 memorization-to-generalization plateau ratio**.
- This is the clearest demonstration I know of why mechanistic interpretability is necessary: the model's actual algorithm was invisible from the outside.

## Biomedical Knowledge Graph Link Prediction — Healthcare AI
**CS 5100 · github.com/Itsme-aniketghosh/biomedical-link-prediction**
- Engineered graph features (PageRank, structural metrics) on the **BioRED** corpus to train a Random Forest link-prediction model hitting **0.94 ROC-AUC** — a **23%+ gain** over learned embeddings — with **5-hop path reasoning** for explainable drug-gene-disease relationships.
- Ran a rigorous **classical ML vs. TransE KG-embedding benchmark (0.94 vs. 0.61 ROC-AUC)**, showing hand-crafted graph features dominate learned representations on low-resource biomedical corpora — a practical finding for data-scarce production settings. This work fed directly into the knowledge-graph thinking I now use at Varosync.

## RAG-based Digital Twin (live)
**github.com/Itsme-aniketghosh/LLM-my-digital-twin · live at ag-hosh-my-digital-twin.hf.space**
- This app — an AI that answers as me, grounded in my own documents. Four modes: chat about my work, generate a tailored cover letter, give an honest job-fit read, or pitch how I'd help a specific team.
- **all-MiniLM-L6-v2** embeddings with in-memory cosine retrieval over an editable Markdown knowledge base; generation on **OpenRouter gpt-oss-120b** with a **Llama 3.1 8B** fallback chain. 2–5 second responses, Gradio UI.

## Autonomous Multi-Agent Trading Simulation — Multi-Agent Systems
**github.com/Itsme-aniketghosh/autonomous-traders-mcp**
- Designed and deployed a multi-agent trading simulation with the **OpenAI Agents SDK**: **4 autonomous AI traders** (Warren, George, Ray, Cathie) coordinated across **6 MCP servers** with **44 distinct tools**, each managing a $10k portfolio, consuming live market data via Polygon.io — with a real-time Gradio dashboard for P&L and tracing.

## AI Research Assistant / Deep Research (live)
**github.com/Itsme-aniketghosh/LLM-Deep-Research · live at ag-hosh-deepresearch.hf.space**
- Multi-agent research pipeline: a Planner generates queries, parallel researchers extract insights via asyncio, and a Writer synthesizes journalistic output with real-time streaming.

## AI-Powered Code Security Analyzer
**github.com/Itsme-aniketghosh/LLM-Cybersecurity-Analyzer**
- Full-stack vulnerability detection combining the **OpenAI Agents SDK with Semgrep**: Next.js frontend, FastAPI backend, deployed on Azure + GCP with Terraform IaC. Built for an "AI in Production" course.

## CrewAI Multi-Agent Collection
**github.com/Itsme-aniketghosh/crewai**
- Five complete multi-agent projects, **15+ agents** total: an Engineering Team, a Stock Picker with hierarchical memory, a Multi-LLM Debate (GPT-4o vs. Claude), a Financial Researcher, and Docker-based code execution.

## Intelligent Traffic Sign Detection (published)
- Integrated **YOLOv8 with a hybrid CNN filtering layer** to cut false positives in cluttered Indian driving environments; trained on Berkeley data, fine-tuned on the IIIT Hyderabad dataset. First-author paper at **AISC 2024 (Springer)**.
