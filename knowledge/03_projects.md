# Projects

My two strongest research results are published on LessWrong — function-vector model diffing and probe-guided quantization — and they're covered in the writing notes. Everything else is below.

## Sufficient Cause Disambiguation (SCD) — Mechanistic Interpretability
CS 7180, PhD seminar with Prof. Byron Wallace · github.com/Itsme-aniketghosh/Sufficient-Cause-Disambiguation

I set out to test whether you can trust what a linear probe tells you about a model's internals, and the answer turned out to be no, at least not on its own. On Llama-3-8B, separator tokens reach 1.000 LDA accuracy across all 32 layers and still produce a 0% flip rate under causal patching. A probe can read a feature perfectly while that feature has no causal effect on the output. Probing accuracy and causal relevance are two different things, and plenty of interpretability claims quietly assume they're one.

The pipeline behind it combines LDA, gradient attribution, and causal activation patching, and gets 448× feature compression at 99.6% accuracy with a 100% prediction-flip rate at α=2.

The same machinery went at an LLM resume screener. Counterfactual name swaps on otherwise identical resumes produced a 95.8% prediction-flip rate, and those swaps accounted for 62% of the model's highest-uncertainty Fit decisions. Systematic name-based bias, measured rather than asserted. A public writeup of that part is still on my list.

Causal patching is fiddly to get right, and a negative result is exactly the kind of finding that's easy to fumble and important not to.

## Grokking and Fourier Features in Neural Networks — Mechanistic Interpretability
Independent research, reproducing Nanda et al. (ICLR 2023) · github.com/Itsme-aniketghosh/grokking-fourier-circuits

Reproduced the grokking result and reverse-engineered what the model had actually learned: a 1-layer transformer trained on modular addition implements a discrete Fourier transform internally. Nobody would guess that from watching its inputs and outputs. Measured an 18:1 memorization-to-generalization plateau ratio.

It's the cleanest argument I know for why mechanistic interpretability is necessary at all. The algorithm was invisible from the outside.

## Biomedical Knowledge Graph Link Prediction — Healthcare AI
CS 5100 · github.com/Itsme-aniketghosh/biomedical-link-prediction

Engineered graph features (PageRank, structural metrics) on the BioRED corpus to train a Random Forest link predictor at 0.94 ROC-AUC, a 23%+ gain over learned embeddings, with 5-hop path reasoning so the drug-gene-disease relationships stay explainable.

The benchmark against TransE embeddings is the interesting half: 0.94 versus 0.61. Hand-crafted graph features beat learned representations on low-resource biomedical corpora, which is worth knowing before you reach for the fancier method in a data-scarce setting. This is also where the knowledge-graph thinking I now use at Varosync came from.

## RAG-based Digital Twin (live)
github.com/Itsme-aniketghosh/LLM-my-digital-twin · live at ag-hosh-my-digital-twin.hf.space

This app. An AI that answers as me, grounded in my own documents, with four modes: chat about my work, draft a tailored cover letter, give an honest job-fit read, or lay out how I'd help a specific team.

all-MiniLM-L6-v2 embeddings with in-memory cosine retrieval over an editable Markdown knowledge base. Generation runs on OpenRouter gpt-oss-120b with a Llama 3.1 8B fallback chain. Responses land in 2–5 seconds, Gradio UI.

## Autonomous Multi-Agent Trading Simulation — Multi-Agent Systems
github.com/Itsme-aniketghosh/autonomous-traders-mcp

Four autonomous AI traders (Warren, George, Ray, Cathie) built on the OpenAI Agents SDK, coordinated across 6 MCP servers with 44 distinct tools, each managing a $10k portfolio off live Polygon.io market data, with a real-time Gradio dashboard for P&L and tracing.

## AI Research Assistant / Deep Research (live)
github.com/Itsme-aniketghosh/LLM-Deep-Research · live at ag-hosh-deepresearch.hf.space

A multi-agent research pipeline: a Planner writes the queries, parallel researchers pull insights via asyncio, and a Writer synthesizes it into journalistic output with real-time streaming.

## AI-Powered Code Security Analyzer
github.com/Itsme-aniketghosh/LLM-Cybersecurity-Analyzer

Full-stack vulnerability detection pairing the OpenAI Agents SDK with Semgrep. Next.js frontend, FastAPI backend, deployed on Azure and GCP with Terraform. Built for an "AI in Production" course.

## CrewAI Multi-Agent Collection
github.com/Itsme-aniketghosh/crewai

Five multi-agent projects, 15+ agents in total: an Engineering Team, a Stock Picker with hierarchical memory, a Multi-LLM Debate (GPT-4o against Claude), a Financial Researcher, and Docker-based code execution.

## Intelligent Traffic Sign Detection (published)
YOLOv8 with a hybrid CNN filtering layer to cut false positives in cluttered Indian driving conditions, trained on Berkeley data and fine-tuned on the IIIT Hyderabad dataset. First-author paper at AISC 2024 (Springer).
