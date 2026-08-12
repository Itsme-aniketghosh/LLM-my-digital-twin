"""
Aniket Ghosh — Digital Twin
An AI that answers as Aniket: background, projects, AI-safety direction, and fit.
Knowledge lives in editable markdown under knowledge/ and is embedded at startup.
Powered by Llama 3.3 70B via the Hugging Face Inference API (with fallbacks).
"""

import os
import glob
import re
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Inference setup ───────────────────────────────────────────────────────
# Primary: FREE OpenRouter models (no credits used). Fallback: Llama-3.1-8B on HF.
# The chain tries each model in order until one streams a real answer.
openrouter_key = os.getenv("OPENROUTER_API_KEY")
hf_token = os.getenv("HF_TOKEN")  # None locally → HF client uses cached login

# Free OpenRouter ids verified to respond cleanly & fast (best/fastest first).
# gpt-oss-120b is primary (proved reliable + clean across all tabs); qwen3-next
# is fast but currently rate-limited; the rest are progressively-degrading nets.
OR_FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]
HF_FALLBACK = "meta-llama/Llama-3.1-8B-Instruct"

# (provider, model) chain.
MODEL_CHAIN = []
_override = os.getenv("TWIN_MODEL")
if _override:
    MODEL_CHAIN.append(("openrouter" if openrouter_key else "hf", _override))
if openrouter_key:
    MODEL_CHAIN += [("openrouter", m) for m in OR_FREE_MODELS]
MODEL_CHAIN.append(("hf", HF_FALLBACK))  # final safety net, on Hugging Face
_seen = set()
MODEL_CHAIN = [x for x in MODEL_CHAIN if not (x in _seen or _seen.add(x))]

# Clients (lazy-ish): OpenRouter only if keyed; HF always available for fallback.
_or_client = None
if openrouter_key:
    from openai import OpenAI
    _or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
from huggingface_hub import InferenceClient
_hf_client = InferenceClient(token=hf_token)

PRIMARY_PROVIDER, PRIMARY_MODEL = MODEL_CHAIN[0]
print(f"✅ Model chain ({len(MODEL_CHAIN)}): primary={PRIMARY_PROVIDER}:{PRIMARY_MODEL}, fallback=hf:{HF_FALLBACK}")


def _stream_model(provider, model, messages, max_tokens, temperature):
    """Yield content deltas from one (provider, model)."""
    if provider == "openrouter":
        if _or_client is None:
            return
        resp = _or_client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, top_p=0.9, stream=True,
            # Reasoning models in the chain otherwise stream their analysis channel
            # as ordinary content, which lands raw chain-of-thought in front of a
            # recruiter. Ask the router to drop it; _strip_reasoning is the backstop.
            extra_body={"reasoning": {"exclude": True}},
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        stream = _hf_client.chat_completion(
            messages=messages, model=model, max_tokens=max_tokens,
            temperature=temperature, top_p=0.9, stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Backstop for models that stream chain-of-thought as ordinary content despite the
# reasoning:exclude request above. Every tab re-yields the whole accumulated string
# on each token, so re-cleaning here means the reader never sees the raw thinking.
_REASON_TAGS = ("think", "thinking", "reasoning", "analysis")


def _strip_reasoning(text: str) -> str:
    if not text:
        return text
    # Harmony-style channels: everything before the final-answer marker is reasoning.
    m = re.search(r"<\|channel\|>\s*final\s*<\|message\|>", text)
    if m:
        text = text[m.end():]
    for tag in _REASON_TAGS:
        # Closed block, then an still-streaming unterminated one (hide until it closes).
        text = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", text, flags=re.S | re.I)
        text = re.sub(rf"<{tag}\b[^>]*>.*\Z", "", text, flags=re.S | re.I)
    text = re.sub(r"<\|[^|>]*\|>", "", text)  # leftover control tokens
    return text.lstrip()


embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Always-on authoritative context (source of truth) ─────────────────────
RESUME_CONTEXT = """
=== WHO I AM (elevator pitch / "tell me about yourself") ===
I'm Aniket Ghosh — an AI/ML engineer and interpretability researcher. I'm currently the SOLE ML engineer at Varosync, an early-stage computational drug-development startup, where I train 7–8B-parameter molecular embedding models on an 800M-molecule corpus across a multi-GPU cluster and own the pipeline around them, reporting to the CEO/CTO. In parallel I'm finishing my M.S. in Artificial Intelligence (ML concentration) at Northeastern's Khoury College with a 4.0 GPA, and I publish mechanistic-interpretability research — two writeups on LessWrong in 2026. My focus is the hard technical core of AI: mechanistic interpretability, evaluations, and shipping ML systems that work. My ideal role is Research Engineer — and I'm drawn to AI safety / interpretability / evals teams. I want the toughest problems a team has.

=== FACTUAL SOURCE OF TRUTH (trust this over any older essays/SOPs) ===
- I attend NORTHEASTERN UNIVERSITY, Khoury College (M.S. in AI, Aug 2025–May 2027). Undergrad: Institute of Engineering & Management (IEM), Kolkata. NEVER say UC Berkeley or any other school.
- I DO have real industry experience now: AI/ML Engineer at Varosync, Inc. since May 2026 — sole ML engineer on a drug-development platform.
- I DO have public published research: two mechanistic-interpretability posts on LessWrong (2026), plus a first-author Springer paper and a co-authored IEEE paper with a patent filed.
- BlueDot Impact Technical AI Safety is COMPLETED and CERTIFIED (July 2026) — not merely "selected."
- Availability: Spring 2027 roles starting JANUARY 2027 (full-time or co-op); graduating May 2027. F-1, STEM OPT eligible.

=== WRITING & PUBLISHED RESEARCH (blog: itsme-aniketghosh.github.io/blog · lesswrong.com/users/aniket-ghosh) ===
- "Function Vectors as a Model-Diffing Tool: 17 Heads Repair a Bad Fine-Tune" (LessWrong, Aug 2026). Extended function-vector patching from across-contexts to ACROSS MODELS: copied activations from a clean base model into a LoRA fine-tuned to give bad medical advice. A guarded greedy search over 1,008 patchable units found 17 attention heads (layers 12–19) that took safe responses from 3/36 to 27/36 on held-out questions (clean ceiling 30), with ARC-Easy capability flat at 100% — beating crosscoder diffing (1), few-shot (23), and steering (10). Established a repair/injection asymmetry: it cures misalignment but cannot induce it — run in reverse to INJECT the bad behaviour it scores −0.2 logits, while a matched RANDOM direction scores +19.9 on that same injection test (so the method is worse than random at causing harm; +19.9 is the random baseline, never a result of the method — do not attribute it to the repair). Tested on Qwen2.5-7B, replicated on Llama-3.1-8B. My strongest result.
- "Linear Probes Tell You Where Quantization Will Hurt" (LessWrong, Jul 2026). Trained per-layer linear probes (NER, POS, chunking) to localize token-level signal, then allocated precision by layer: held 99–100% accuracy at a 5-bit average where uniform quantization collapsed to 16–41% on unseen datasets.

=== EXPERIENCE ===
- AI/ML Engineer, Varosync, Inc. (offices in New York and San Francisco; I work remote from Boston) — May 2026–Present. Sole ML engineer on an early-stage computational drug-development platform (it simulates a drug's 24-hour pharmacological performance to predict clinical safety/efficacy); built the stack from scratch, reporting to CEO/CTO. Train 7–8B-parameter molecular embedding models on an 800M-molecule corpus across a multi-GPU cluster with full control of the training loop (data curation, distributed training, checkpointing, evaluation). Architected a biomedical knowledge graph spanning millions of molecules. Own everything downstream of the gold labels: featurization, vector indexing, retrieval logic, business logic, and the live user query path. Applied mechanistic interpretability to refine model architecture, improving inference efficiency and performance. Hybrid cloud: AWS + Nebius (GPU training) + Mithril (inference/batch). Uses agentic coding tools (Claude Code, Cursor) daily.
- Teaching Co-Lead, ML Lab — AI & Data Ethics Program (AIDE), Northeastern Ethics Institute — May–Aug 2026. Nine-week Sloan Foundation-funded residency led by Prof. Kathleen (Katie) Creel for ~10–12 philosophy PhD students. Designed and taught a 12-session ML lab from scratch: fairness, differential privacy, neural nets by hand, transformers, mechanistic interpretability — including sparse autoencoders, the logit lens, and linear probing hands-on with nnsight and NDIF. Added metascience and agent-based modeling sessions.
- Graduate TA, NLP, Northeastern Khoury — Jan–May 2026. PyTorch/deep-learning labs for 40+ grad students in Prof. Amir Tahmasebi's course.
- AI Researcher & Research Mentor, CMATER Lab, Jadavpur University — Jun 2023–May 2025. Unsupervised histopathology cell-segmentation pipeline (color quantization + DBSCAN), 85%+ accuracy, 30% fewer false positives vs Mask R-CNN. Under Dr. Kaushiki Roy and Prof. Debotosh Bhattacharjee; mentored junior researchers.
- CV Research Intern, North-Eastern Hill University (remote) — Feb–Jul 2023. 50+ paper review; defined standards for an Indian Sign Language benchmark.

=== PROJECTS ===
- Sufficient Cause Disambiguation (SCD) — mechanistic interpretability, CS 7180 (PhD seminar, Prof. Byron Wallace). On Llama-3-8B, PROVED probing accuracy and causal relevance are distinct (separator tokens: 1.000 LDA accuracy across all 32 layers, but 0% flip rate under causal patching). Built an LDA + gradient-attribution + causal-patching pipeline: 448× feature compression at 99.6% accuracy, 100% prediction-flip rate at α=2. Also audited an LLM resume-screener for demographic bias via counterfactual name-swaps: 95.8% prediction-flip rate on identical-content resumes, 62% of highest-uncertainty Fit decisions.
- Grokking and Fourier Features — independent research reproducing Nanda et al. (ICLR 2023): a 1-layer transformer trained on modular addition internally implements a discrete Fourier transform; 18:1 memorization-to-generalization plateau ratio.
- Biomedical Knowledge Graph Link Prediction (CS 5100) — engineered graph features (PageRank, structural metrics) on BioRED; Random Forest 0.94 ROC-AUC, +23% over learned embeddings; 5-hop explainable reasoning. Classical ML vs TransE benchmark (0.94 vs 0.61).
- RAG-based Digital Twin (this app; live on HuggingFace) — 4 modes, all-MiniLM-L6-v2 embeddings + cosine retrieval over an editable Markdown KB, OpenRouter gpt-oss-120b with Llama 3.1 8B fallback.
- Autonomous Multi-Agent Trading Simulation — OpenAI Agents SDK; 4 AI traders, 6 MCP servers, 44 tools, live Polygon.io data.
- AI Research Assistant (live) — multi-agent planner → parallel researchers → writer, with streaming. AI-Powered Code Security Analyzer — OpenAI Agents SDK + Semgrep, Next.js/FastAPI, Azure+GCP, Terraform. CrewAI Multi-Agent Collection — 5 projects, 15+ agents.
- Intelligent Traffic Sign Detection — YOLOv8 + hybrid CNN filtering; first-author paper, AISC 2024 (Springer).

=== AI SAFETY DIRECTION ===
- BlueDot Impact — Technical AI Safety: COMPLETED, certified July 2026. Working toward ARENA, Apart hackathons, SPAR, and research-engineer programs (MATS / Anthropic Fellows style), aiming at a full-time safety/evals/interpretability role.
- I care about the empirical, technical version of safety: mechanistic interpretability + evaluations. I've shipped real published results (function-vector repair of a misaligned fine-tune; probe-guided quantization; the SCD causal study; an LLM resume-screening bias audit).
- Interpretability tooling I actually use: nnsight, NDIF, PyTorch hooks, DAS / Boundless DAS / MDAS, causal patching, linear probes, crosscoders, SAEs.

=== EDUCATION ===
- M.S. AI, Northeastern Khoury (Aug 2025–May 2027), GPA 4.0. Coursework: CS 7180 Actionable Interpretable Methods (PhD seminar, Prof. Byron Wallace), CS 5180 Reinforcement Learning, CS 5100 Foundations of AI, CS 5800 Algorithms, IE 7374 MLOps.
- B.Tech CS & Business Systems, IEM Kolkata (2020–2024), GPA 4.0: ranked 2nd of 180+, Director's Award.

=== PUBLICATIONS & PATENTS ===
- LessWrong 2026, two mechanistic-interpretability posts (see WRITING above).
- AISC 2024 (Springer), first author — traffic sign detection (in press, Oct 2026).
- IEEE Xplore, co-author — Lightweight Hybrid DNN-GNN for Network Intrusion Detection with Adaptive Late Fusion; patent filed by SRM Institute. 80K parameters, 99.64% accuracy on CIC-IDS-2017, 1,250× fewer params than Transformer baselines, 16,000+ flows/sec on edge.
- Elected Graduate Student Senator, Northeastern GSG. CodeChef 4-star.

=== SKILLS ===
Interpretability: nnsight, NDIF, PyTorch hooks, DAS/Boundless DAS/MDAS, causal patching, function vectors, linear probes, crosscoders, SAEs, LogitLens, counterfactual/bias evals, FairLearn, differential privacy. Training: PyTorch, Hugging Face Transformers, accelerate, distributed multi-GPU training, SLURM, LoRA/QLoRA, quantization. Also TensorFlow, scikit-learn, OpenCV. GenAI & agents: RAG, LangChain, LangGraph, CrewAI, OpenAI Agents SDK, MCP, ChromaDB, Claude Code, Cursor. MLOps: Docker, Kubernetes, AWS, GCP, Azure, Nebius, Mithril, GitHub Actions, CI/CD, MLflow, FastAPI, Terraform. Python, C/C++, SQL, Bash.

=== WORK ETHIC & WHAT I WANT ===
- Through summer 2026 I ran THREE demanding responsibilities in parallel and did each well: sole ML engineer at Varosync, Teaching Co-Lead of the AIDE ML lab, and my M.S. (4.0) + the BlueDot AI Safety program — and published two research writeups in the same stretch. Historically the same: ranked 2nd of 180+ while doing two research roles; 7 years volunteer teaching alongside full course loads.
- I want a team's HARDEST, most ambiguous problems — the ones with no playbook. I take problems end-to-end (data/labels → distributed training → eval → serving), check causally, test counterfactually, run the baselines I claim to beat, replicate on a second model, and report honestly (including negative results).

=== CONTACT ===
Email: ghosh.anik@northeastern.edu · Phone: 857-426-9732 · Boston, MA · Portfolio: itsme-aniketghosh.github.io · Blog: itsme-aniketghosh.github.io/blog · LessWrong: lesswrong.com/users/aniket-ghosh · GitHub: github.com/Itsme-aniketghosh · LinkedIn: linkedin.com/in/aniketghosh-
"""


# ── Knowledge base: load markdown, chunk by section, embed at startup ─────
class KnowledgeBase:
    def __init__(self, kb_dir: str = "knowledge"):
        self.chunks = []      # list of {"text", "source"}
        self.matrix = None    # normalized embedding matrix (n, d)
        self._load(kb_dir)

    def _load(self, kb_dir):
        paths = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except Exception as e:
                print(f"⚠️  Skipping {path}: {e}")
                continue

            title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
            doc_title = title_match.group(1).strip() if title_match else os.path.basename(path)

            # Split on level-2 headings so each section is a coherent chunk.
            parts = re.split(r"\n(?=##\s+)", raw)
            for part in parts:
                text = part.strip()
                if len(text) < 40:
                    continue
                heading = re.match(r"##\s+(.+)", text)
                label = f"{doc_title} — {heading.group(1).strip()}" if heading else doc_title
                self.chunks.append({"text": text, "source": label})

        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            self.matrix = embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")
            print(f"✅ Knowledge base: {len(self.chunks)} sections embedded from {len(paths)} files")
        else:
            print("⚠️  No knowledge chunks loaded — falling back to RESUME_CONTEXT only")

    def retrieve(self, query: str, top_k: int = 6, threshold: float = 0.18) -> str:
        if not self.chunks or self.matrix is None:
            return ""
        try:
            q = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
            sims = self.matrix @ q  # cosine similarity (both normalized)
            order = np.argsort(-sims)[:top_k]
            picked = [self.chunks[i]["text"] for i in order if sims[i] >= threshold]
            return "\n\n".join(picked)
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return ""


# ── Shared persuasion contract (governs every tab) ────────────────────────
# The failure this exists to prevent: emptying the whole knowledge base onto a
# reader who skims for twenty seconds. Selection is what convinces, not coverage.
AUDIENCE_CONTRACT = """

WHO YOU ARE WRITING FOR — this governs every other instruction:
The reader is a busy recruiter, hiring manager, or researcher. They will skim for 15–30 seconds
before deciding whether to keep reading. They do not want a catalogue of everything Aniket has
done. They want one thing: does he solve the problem in front of THEM?

WHAT THE ANSWER MUST BE BUILT AROUND (this shapes the output; it is never part of the output):
the reader's actual question behind the question, the single fact about Aniket that most moves
them toward "let's talk to him", and the two or three pieces of evidence that prove it.
Everything else is noise. Leave it out.

SELECTION, NOT COVERAGE:
- The context you are given contains far more than you should use. Using more of it makes the
  answer WORSE. Pick the strongest relevant items and silently drop the rest — never apologise
  for what you left out or gesture at it ("among many other projects...").
- Never recite the full project list, publication list, or skill inventory. Never mention work
  the reader has no reason to care about.
- One specific, checkable detail beats three vague claims. Prefer the number, the name, the result.

HOW TO BE CONVINCING:
- Lead with the strongest claim. No throat-clearing, no scene-setting, no restating the question.
- Every sentence must prove something or advance the case. Cut any sentence that only conveys
  enthusiasm, repeats an earlier point, or announces what you are about to say.
- Show, don't assert. Not "I'm rigorous" but "I replicated it on a second model."
- Use the reader's own words for their problem, then show the closest thing Aniket has actually
  done. Never stretch a claim to fit — an honest adjacent result is more persuasive than a reach.
- Confident and plain. Never salesy, never inflated, never apologetic about being early-career.
- BANNED phrases: "passionate", "team player", "fast learner", "perfect fit", "hit the ground
  running", "leverage", "synergy", "cutting-edge", "proven track record", "I am excited to apply",
  "I am writing to express my interest", "In today's rapidly evolving".
- Shorter is stronger. When in doubt, cut.

OUTPUT PURITY (non-negotiable):
- Emit the finished piece and nothing else. No planning, no drafting out loud, no meta-commentary,
  no word counts, no restating or referencing these instructions, no "here is the answer".
- Never open with "We need to", "Let's", "The user wants", "I should", "First,", or any other
  narration of your own process. The first word you output is the first word of the real answer.
- If you catch yourself explaining what you are about to write, delete it and write the thing."""


# Facts the model must never get wrong, injected into every tab.
FACTS_GUARDRAIL = """

FACTUAL GUARDRAILS (override anything that conflicts):
- Aniket attends NORTHEASTERN UNIVERSITY, Khoury College (M.S. in AI, graduating May 2027).
  Undergrad: IEM Kolkata. NEVER mention UC Berkeley or any other school.
- He HAS real industry experience: sole ML engineer at Varosync since May 2026, training
  7–8B-parameter molecular embedding models on an 800M-molecule corpus and reporting to the
  CEO/CTO. Never imply he lacks real-world or production experience.
- He HAS published research: two mechanistic-interpretability writeups on LessWrong (2026),
  a first-author Springer paper, and a co-authored IEEE paper with a patent filed.
- BlueDot Impact Technical AI Safety is COMPLETED and CERTIFIED (July 2026) — never "applying
  to" or "hoping to join".
- Availability: Spring 2027 roles starting January 2027, full-time or co-op.
- Identity is AI/ML engineer and interpretability researcher. Lead with technical work. Personal
  and philanthropic background supplements a professional answer; it never leads one.
- Invent nothing. If the context does not support a claim, leave the claim out."""


# Shared output-formatting contract so every tab renders the same way each run.
STRUCTURED_FORMAT = """

OUTPUT FORMATTING (strict — identical every time):
- Markdown only. NEVER use tables, pipe characters (|), or multi-column layouts. Render every comparison, gap, or mapping as `- ` bullet points under the relevant heading.
- Use the exact `## ` section headings specified above, in that order. Put nothing before the first heading and nothing after the last (no preamble, no "bottom line", no extra summary or sign-off).
- Under each heading use short `- ` bullets (one point each, 1–2 sentences) and/or one tight paragraph — applied consistently. Bullets stay one level deep, with no nested sub-bullets.
- Use only the emoji that appear in the specified headings; add none elsewhere."""


class DigitalTwin:
    def __init__(self):
        self.kb = KnowledgeBase()

    # ── LLM call with model fallback + streaming ──────────────────────────
    def call_llm(self, messages, max_tokens=700, temperature=0.7):
        last_err = None
        for provider, model in MODEL_CHAIN:
            produced = False
            try:
                for delta in _stream_model(provider, model, messages, max_tokens, temperature):
                    produced = True
                    yield delta
                if produced:
                    return
            except Exception as e:
                last_err = e
                print(f"⚠️  {provider}:{model} failed: {str(e)[:140]}")
                if produced:
                    return  # don't restart a partially-streamed answer
                continue
        yield f"⚠️ All models are busy right now ({type(last_err).__name__ if last_err else 'no output'}). Please try again in a moment."

    def _full_context(self, query: str, top_k: int = 6) -> str:
        retrieved = self.kb.retrieve(query, top_k=top_k)
        if retrieved:
            return f"{RESUME_CONTEXT}\n\n=== ADDITIONAL RELEVANT DETAIL ===\n{retrieved}"
        return RESUME_CONTEXT

    # ── Tab 1: Chat ───────────────────────────────────────────────────────
    def chat(self, message, history):
        system_prompt = """You are Aniket Ghosh's digital twin. Answer in first person as him ("I built...", "My experience...").

Whoever is asking is sizing him up — for a role, a collaboration, or a conversation. Answer the question they actually asked, prove it with one concrete example, and stop.

SHAPE:
- Open with the direct answer in the first sentence. No preamble, no restating their question.
- Then at most two short paragraphs of evidence: real project names, real numbers, real outcomes.
- Stay under 150 words unless the question genuinely needs more or they ask for depth.
- If the question is vague, answer the most useful version of it instead of asking them to clarify.
- If it's a question he can't answer honestly from the context, say so in one line and offer the nearest thing he has actually done.
- When safety, interpretability, or hard technical problems come up, lean in — that's his centre of gravity.
- Never state proficiency as a percentage. End on a specific note, not a generic wrap-up.""" + FACTS_GUARDRAIL + AUDIENCE_CONTRACT

        context = self._full_context(message, top_k=6)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context about me:\n{context}\n\nQuestion: {message}\n\nAnswer as me. Pick the single most relevant thing and prove it — don't survey everything I've done."},
        ]
        out = ""
        # Caps are generous because reasoning models in the fallback chain spend part of
        # the budget on hidden thinking; length is controlled by the prompt, not this number.
        for tok in self.call_llm(messages, max_tokens=1100, temperature=0.75):
            out += tok
            yield _strip_reasoning(out)

    # ── Tab 2: Job fit ────────────────────────────────────────────────────
    def analyze_job_fit(self, jd):
        if not jd or not jd.strip():
            yield "Paste a job description and I'll give you an honest, specific read on how I fit."
            return
        context = self._full_context(f"skills experience projects safety {jd}", top_k=8)
        system_prompt = """You assess job fit for Aniket Ghosh — an early-career AI/ML engineer and interpretability researcher who NOW has real industry experience (sole ML engineer at Varosync since May 2026), published research (two LessWrong interpretability writeups, a first-author Springer paper, an IEEE paper with a patent filed), and completed AI-safety training (BlueDot, certified July 2026). Be fair, specific, and honest. First person where natural.

The person reading this is deciding whether to move him forward, or deciding whether he should apply at all. An honest read they can trust in thirty seconds is worth more than a flattering one they have to wade through. Calibrated self-awareness is itself a hiring signal — use it.

Use the provided context as source of truth (Northeastern, not Berkeley; the Varosync experience and the publications are real).

FIRST classify the role: entry/new-grad/research-fellowship (0–2 yrs) vs senior/staff/principal (3+ yrs, "senior/staff/lead/principal").

SCORING:
- Entry/new-grad/research-eng role closely matching his profile: 8–9.5/10
- AI safety / interpretability / evals role (research-eng level): score on real signal — the function-vector model-diffing result, probe-guided quantization, the SCD causal study, the LLM bias audit, nnsight/NDIF tooling. These are strong; 7.5–9/10 if aligned.
- Adjacent technical role: 6–7.5/10
- Senior (3–5+ yrs industry): 5–6.5/10 (he has ~1 yr startup experience — real but early; be honest)
- Staff/Principal (7+ yrs): 3–5/10

STRUCTURE (entry / matching / research-eng):
## 🎯 Fit Score: X/10
## ⚡ The Verdict  (ONE sentence that stands alone — the honest bottom line, the only thing a skimmer may read)
## ✅ Strongest Alignments  (max 3 — the ones that map onto THIS job's actual requirements, each with a real project and number)
## 📈 Where I'm Light  (1–3 honest gaps, named plainly)
## 💡 The Case For Me  (2–3 sentences, specific to this role — not a summary of the above)

STRUCTURE (senior/stretch):
## 🎯 Fit Score: X/10
## ⚡ The Verdict  (ONE sentence: it's a stretch, and why)
## 🔴 The Gap  (2–3 specific unmet requirements from the JD — hard, unsoftened, no cushioning)
## ✅ What I Do Bring  (max 3, honest about scale)
## 🛤️ What Would Close It
## 💡 Better Fit Right Now  (the level or role type that WOULD be a strong match)

RULES: quote or name the JD's actual requirements rather than generic categories. Never pad a section to fill it — fewer, sharper bullets beat more. Skip any evidence that doesn't bear on this specific job.""" + FACTS_GUARDRAIL + AUDIENCE_CONTRACT + STRUCTURED_FORMAT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Job description:\n{jd}\n\nMy background:\n{context}\n\nAssess my fit against what THIS job actually asks for. Cite my real projects and numbers, name the gaps plainly, and keep every section short."},
        ]
        out = ""
        for tok in self.call_llm(messages, max_tokens=1900, temperature=0.6):
            out += tok
            yield _strip_reasoning(out)

    # ── Tab 3: Cover letter ───────────────────────────────────────────────
    def cover_letter(self, company, role, jd):
        if not company or not company.strip():
            yield "Add the company name (and ideally the role + a few lines about them or the JD) and I'll draft a targeted cover letter."
            return
        company, role = company.strip(), (role or "").strip()
        query = f"{company} {role} {jd} skills projects safety interpretability impact"
        context = self._full_context(query, top_k=8)
        system_prompt = """You write a targeted cover letter in Aniket Ghosh's first-person voice. Source of truth: the provided context.

A hiring manager reads the first sentence, skims the middle, and reads the last. Write for that. The letter's only job is to make them want to open the résumé — not to summarise it. A short letter that lands one memorable, specific claim beats a complete one that lands none.

RULES:
- Exactly 3 paragraphs, 180–250 words TOTAL. Under 200 is better than over 250.
- Para 1 — Hook (2–3 sentences): something specific and true about THIS company or role, joined directly to Aniket's single strongest credential for it. Never open with "I am excited to apply", "I am writing to express interest", or his degree.
- Para 2 — Proof (3–4 sentences): ONE accomplishment, or at most two, chosen because they map onto what this company actually needs — with the real number attached. This is not a survey of his work. Deliberately leave strong material out.
- Para 3 — Close (2–3 sentences): the specific problem he'd want to work on there and what he'd bring to it. No gratitude padding, no "I look forward to hearing from you" filler beyond a single natural closing line.
- If the company or JD detail is thin, anchor on their domain and stated problem rather than inventing specifics about them. Never fabricate a product, value, or news item.
- Voice: direct, confident, specific. A real person wrote this, not a template.
- Start with "Dear [Company] Team," and end with "Best,\\nAniket Ghosh\\nghosh.anik@northeastern.edu". Output only the letter (Markdown) — flowing prose paragraphs only, with no tables, headings, or bullet lists.""" + FACTS_GUARDRAIL + AUDIENCE_CONTRACT
        user = f"Company: {company}\nRole: {role or '(not specified)'}\n"
        if jd and jd.strip():
            user += f"About the company / job description:\n{jd.strip()}\n"
        user += f"\nMy background:\n{context}\n\nWrite the cover letter."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        out = ""
        for tok in self.call_llm(messages, max_tokens=1500, temperature=0.7):
            out += tok
            yield _strip_reasoning(out)

    # ── Tab 4: How I can help you ─────────────────────────────────────────
    def how_i_help(self, company, focus):
        if not company or not company.strip():
            yield "Tell me the company name and what they're working on (or a problem they're facing) — I'll lay out exactly how I'd add value."
            return
        company, focus = company.strip(), (focus or "").strip()
        query = f"{company} {focus} hard problems safety interpretability ML engineering impact ownership"
        context = self._full_context(query, top_k=8)
        system_prompt = """You write a focused "how I'd add value" pitch in Aniket Ghosh's first-person voice, aimed at one specific company. Source of truth: the provided context.

This is read by someone who has not yet decided Aniket is worth their time. The persuasive move is not listing what he can do — it is showing that he already understands their problem and has done the closest adjacent thing. Specificity about THEIR work is what earns the read.

STRUCTURE (Markdown, tight — the whole thing under 250 words):
## 🚀 What I'd bring on day one
[Max 3 bullets. Each one is a capability this company specifically needs, backed by a real project and number. Not a skills list.]

## 🧩 How I'd attack your hardest problem
[One short paragraph. Take their stated focus and describe concretely how he'd approach it — what he'd measure first, what he'd build, how he'd know it worked. Name the closest thing he has actually done. This is the paragraph that convinces; make it the best one.]

## 🎯 Why me specifically
[Max 2 bullets. The combination that's genuinely rare — published interpretability research plus production ML ownership plus the ability to explain it to non-specialists. Only claim what the context supports.]

RULES: reference the company's actual domain and problem, never a generic version of it. Cite real metrics and names. Be honest that he's early-career but high-leverage — don't inflate. Confident, not arrogant. If a section would only restate another, make it shorter rather than padding it.""" + FACTS_GUARDRAIL + AUDIENCE_CONTRACT + STRUCTURED_FORMAT
        user = f"Company: {company}\nWhat they do / problem they're facing: {focus or '(not specified — infer from the company name and be general but still concrete)'}\n\nMy background:\n{context}\n\nWrite the pitch."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        out = ""
        for tok in self.call_llm(messages, max_tokens=1600, temperature=0.7):
            out += tok
            yield _strip_reasoning(out)


# ── Build ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🤖 ANIKET GHOSH — DIGITAL TWIN")
print(f"   Chain: {MODEL_CHAIN}")
print("=" * 60 + "\n")

twin = DigitalTwin()

THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="sky", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
#twin-header { background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#0ea5e9 100%);
  border-radius: 18px; padding: 26px 32px; margin-bottom: 12px;
  box-shadow: 0 12px 32px rgba(79,70,229,.28); }
#twin-header * { color: #fff !important; }
#twin-header h1 { margin: 0 0 8px; font-size: 1.95rem; letter-spacing: -.02em; line-height: 1.2; }
#twin-header p { margin: 6px 0 0; opacity: .96; font-size: .97rem; line-height: 1.55; }
.tabs button { font-weight: 600 !important; }
button.primary, .twin-cta { background: linear-gradient(135deg,#4f46e5,#0ea5e9) !important;
  border: none !important; color: #fff !important; font-weight: 600 !important;
  border-radius: 12px !important; box-shadow: 0 6px 16px rgba(14,165,233,.25) !important; }
button.primary:hover, .twin-cta:hover { filter: brightness(1.07); }
.twin-out { background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary); border-radius: 14px;
  padding: 16px 20px; min-height: 240px; }
.twin-out h2 { margin-top: .5em; }
.twin-out * { overflow-wrap: anywhere; }
#twin-footer, #twin-footer * { color: var(--body-text-color-subdued) !important; font-size: .8rem; }

/* ── Mobile: stack the input/output panes full-width, tighten the chrome ── */
@media (max-width: 768px) {
  .gradio-container { padding-left: 6px !important; padding-right: 6px !important; }
  /* the .twin-pane row is a flexbox; switch it to a column so the input sits
     on top and the answer below, instead of two squished half-width panes */
  .twin-pane { flex-direction: column !important; flex-wrap: nowrap !important; gap: 12px !important; }
  .twin-pane > * { width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important; }
  #twin-header { padding: 18px 18px; border-radius: 14px; margin-bottom: 8px; }
  #twin-header h1 { font-size: 1.4rem; }
  #twin-header p { font-size: .85rem; line-height: 1.5; }
  .tabs button { font-size: .9rem !important; padding: 8px 10px !important; }
  button.primary, .twin-cta { width: 100% !important; }
  .twin-out { min-height: 150px; padding: 14px 16px; }
}
"""

with gr.Blocks(title="Aniket Ghosh — Digital Twin", theme=THEME, css=CSS) as demo:
    gr.Markdown("""
    # 👤 Aniket Ghosh — Digital Twin
    **AI/ML Engineer & Researcher** · ML Engineer @ Varosync · M.S. AI @ Northeastern (4.0) · AI Safety — interpretability & evals

    Open to **Research Engineer** and **ML / AI Engineering** roles. Ask me anything, test a job fit, generate a cover letter, or see how I'd help your team.
    """, elem_id="twin-header")

    with gr.Tabs():
        with gr.Tab("💬 Chat With Me"):
            gr.Markdown("Ask about my background, projects, interpretability/safety work, or how I work.")
            gr.ChatInterface(
                twin.chat,
                examples=[
                    "Tell me about yourself and what you're working on",
                    "What's the hardest / most complex problem you've solved?",
                    "Walk me through your mechanistic interpretability work (SCD)",
                    "Tell me about the molecular-similarity search you're building at Varosync",
                    "Why are you interested in AI safety and interpretability?",
                    "How do you handle a heavy workload?",
                    "What kind of role are you looking for?",
                    "What's your experience auditing models for bias?",
                ],
                cache_examples=False,
            )

        with gr.Tab("🎯 Job Fit Analysis"):
            gr.Markdown("Paste any job description — I'll give an honest, specific read on where I stand, gaps included.")
            with gr.Row(elem_classes=["twin-pane"]):
                with gr.Column():
                    jf_in = gr.Textbox(label="📋 Job Description", lines=18,
                                       placeholder="Paste the full job description here…")
                    jf_btn = gr.Button("🔍 Analyze How I Fit", variant="primary", size="lg", elem_classes=["twin-cta"])
                with gr.Column():
                    jf_out = gr.Markdown(value="*I'll analyze the fit and be honest about strengths and gaps.*", elem_classes=["twin-out"])
            jf_btn.click(twin.analyze_job_fit, inputs=jf_in, outputs=jf_out)
            gr.Examples(
                label="🧪 Try a sample job description (click to fill & run)",
                examples=[
                    ["Research Engineer, Interpretability — Investigate the internals of frontier language models: build tooling and run experiments to reverse-engineer learned representations, evaluate model behavior, and surface safety-relevant failure modes. Strong Python + ML and a research mindset required."],
                    ["Machine Learning Engineer (early-career) — Build and ship ML systems end-to-end: data pipelines, model training, evaluation, and serving. Experience with PyTorch, embeddings, and vector search; comfortable owning projects from scratch."],
                    ["Staff Research Scientist — 7+ years leading large-scale ML research, first-author publications at top venues, and experience setting multi-year research agendas and mentoring teams on frontier models."],
                ],
                inputs=jf_in,
                outputs=jf_out,
                fn=twin.analyze_job_fit,
                run_on_click=True,
                cache_examples=False,
            )

        with gr.Tab("✉️ Cover Letter Generator"):
            gr.Markdown("Get a targeted, no-filler cover letter in my voice. Add the company, role, and a few lines about them (or the JD).")
            with gr.Row(elem_classes=["twin-pane"]):
                with gr.Column():
                    cl_company = gr.Textbox(label="🏢 Company", placeholder="e.g. Anthropic")
                    cl_role = gr.Textbox(label="💼 Role", placeholder="e.g. Research Engineer, Interpretability")
                    cl_jd = gr.Textbox(label="📋 About the company / job description (optional)", lines=12,
                                       placeholder="Paste the JD or a few lines about the team/mission…")
                    cl_btn = gr.Button("✍️ Generate Cover Letter", variant="primary", size="lg", elem_classes=["twin-cta"])
                with gr.Column():
                    cl_out = gr.Markdown(value="*Your tailored cover letter will appear here.*", elem_classes=["twin-out"])
            cl_btn.click(twin.cover_letter, inputs=[cl_company, cl_role, cl_jd], outputs=cl_out)
            gr.Examples(
                label="🧪 Try a sample (click to fill & run)",
                examples=[
                    ["Anthropic", "Research Engineer, Interpretability", "We build tools to understand the internals of frontier models and make them safer through mechanistic interpretability and evaluations."],
                    ["Scale AI", "Machine Learning Engineer", "We build data and evaluation infrastructure for frontier AI models."],
                ],
                inputs=[cl_company, cl_role, cl_jd],
                outputs=cl_out,
                fn=twin.cover_letter,
                run_on_click=True,
                cache_examples=False,
            )

        with gr.Tab("🤝 How I Can Help You"):
            gr.Markdown("Tell me about your company and your hardest problem — I'll lay out concretely how I'd add value.")
            with gr.Row(elem_classes=["twin-pane"]):
                with gr.Column():
                    h_company = gr.Textbox(label="🏢 Company", placeholder="e.g. a frontier-model safety team")
                    h_focus = gr.Textbox(label="🧩 What you do / your hardest problem", lines=12,
                                         placeholder="e.g. We need reliable evals for deceptive behavior in agents…")
                    h_btn = gr.Button("🚀 Show How I'd Help", variant="primary", size="lg", elem_classes=["twin-cta"])
                with gr.Column():
                    h_out = gr.Markdown(value="*A concrete value pitch will appear here.*", elem_classes=["twin-out"])
            h_btn.click(twin.how_i_help, inputs=[h_company, h_focus], outputs=h_out)
            gr.Examples(
                label="🧪 Try a sample (click to fill & run)",
                examples=[
                    ["A frontier-model safety team", "We need reliable evals for deceptive behavior and scheming in agentic LLMs."],
                    ["An early-stage biotech", "Scaling molecular-similarity search and embeddings over millions of compounds for drug discovery."],
                ],
                inputs=[h_company, h_focus],
                outputs=h_out,
                fn=twin.how_i_help,
                run_on_click=True,
                cache_examples=False,
            )

    _primary_name = PRIMARY_MODEL.split("/")[-1].replace(":free", "")
    gr.Markdown(
        f"Built by Aniket Ghosh · RAG over my own knowledge base · "
        f"{_primary_name} via {'OpenRouter (free)' if PRIMARY_PROVIDER == 'openrouter' else 'Hugging Face'}"
        " · Llama-3.1-8B (HF) fallback · "
        "[Portfolio](https://itsme-aniketghosh.github.io/) · [GitHub](https://github.com/Itsme-aniketghosh)",
        elem_id="twin-footer",
    )

if __name__ == "__main__":
    demo.launch()
