# Projects

## Sufficient Cause Disambiguation (SCD) — Mechanistic Interpretability
**CS 7180 (PhD-level) · github.com/Itsme-aniketghosh/Sufficient-Cause-Disambiguation**
This is my flagship hard-problem and safety-relevant project.
- **Proved probing accuracy and causal relevance are distinct properties** on Llama-3-8B: separator tokens reach 1.000 LDA accuracy across all 32 layers yet drive a **0% flip rate under causal patching**. In plain terms — a probe can read a feature perfectly while that feature has no causal effect on the model's output. That distinction matters a lot for interpretability claims.
- Built a full **mechanistic interpretability pipeline** (LDA + gradient attribution + causal activation patching), achieving **448× feature compression at 99.6% accuracy** and a **100% prediction-flip rate** at α=2.
- **Audited an LLM resume-screener for demographic bias** via counterfactual name-swap perturbations: identical-content resumes triggered a **95.8% prediction-flip rate** and accounted for **62% of the model's highest-uncertainty Fit decisions** — surfacing systematic name-based bias. This is a concrete evals/safety result, and I plan to publish a public writeup.
- Why it's hard: causal patching is fiddly, and the negative result (probing ≠ causation) is the kind of thing that's easy to get wrong and important to get right.

## Biomedical Knowledge Graph Link Prediction — Healthcare AI
**github.com/Itsme-aniketghosh/biomedical-link-prediction**
- Engineered graph features (PageRank, structural metrics) on the **BioRED** corpus to train a Random Forest link-prediction model hitting **0.94 ROC-AUC** — a **23%+ gain** over learned embeddings — with **5-hop reasoning** for explainable drug-gene-disease relationships.
- Ran a rigorous **classical ML vs. TransE KG-embedding benchmark (0.94 vs. 0.61 ROC-AUC)**, showing hand-crafted graph features dominate learned representations on low-resource biomedical corpora — a practical finding for data-scarce production settings. This work fed directly into the knowledge-graph thinking I now use at Varosync.

## Autonomous Multi-Agent Trading Simulation — Multi-Agent Systems
**github.com/Itsme-aniketghosh/autonomous-traders-mcp**
- Designed and deployed a multi-agent trading simulation with the **OpenAI Agents SDK**: **4 autonomous AI traders** coordinated across **6 MCP servers** with **44 distinct tools**, consuming live market data via Polygon.io for end-to-end autonomous portfolio management, with a real-time Gradio dashboard for P&L and tracing.

## Intelligent Traffic Sign Detection (published)
- Integrated **YOLOv8 with a hybrid CNN filtering layer**; trained on Berkeley data, fine-tuned on the IIIT Hyderabad dataset for Indian road conditions. First-author paper at **AISC 2024 (Springer)**.
