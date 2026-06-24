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


embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Always-on authoritative context (source of truth) ─────────────────────
RESUME_CONTEXT = """
=== WHO I AM (elevator pitch / "tell me about yourself") ===
I'm Aniket Ghosh — an AI/ML engineer and researcher. I'm currently the SOLE ML engineer at Varosync, an early-stage drug-discovery startup in Boston, where I'm building a molecular-similarity search system over millions of molecules from scratch and reporting to the CEO/CTO. In parallel I'm finishing my M.S. in Artificial Intelligence (ML concentration) at Northeastern University with a 4.0 GPA. My focus is the hard technical core of AI: mechanistic interpretability, evaluations, and shipping ML systems that work. My ideal role is Research Engineer — and I'm drawn to AI safety / interpretability / evals teams. I want the toughest problems a team has.

=== FACTUAL SOURCE OF TRUTH (trust this over any older essays/SOPs) ===
- I attend NORTHEASTERN UNIVERSITY (M.S. in AI, Aug 2025–May 2027). Undergrad: Institute of Engineering & Management (IEM), Kolkata. NEVER say UC Berkeley or any other school.
- I DO have real industry experience now: AI/ML Engineer at Varosync, Inc. (Boston) since May 2026 — sole ML engineer on a drug-discovery platform.

=== EXPERIENCE ===
- AI/ML Engineer, Varosync, Inc. (Boston) — May 2026–Present. Sole ML engineer on an early-stage drug-discovery platform; built the stack from scratch, reporting to CEO/CTO. Architected a biomedical knowledge graph spanning millions of molecules; building a molecular-similarity search system — training molecular-similarity embedding models over millions of molecules and owning the full pipeline (gold-label curation → featurization → embedding training → evaluation → vector indexing → query layer) on hybrid cloud. Uses agentic coding tools (Claude Code, Cursor) daily.
- AI Researcher, CMATER Lab, Jadavpur University — Jun 2023–May 2025. Unsupervised histopathology cell-segmentation pipeline (color quantization + DBSCAN), 85%+ accuracy, 30% fewer false positives vs Mask R-CNN. Mentored junior researchers.
- CV Research Intern, North-Eastern Hill University (remote) — Feb–Jul 2023. 50+ paper review; defined standards for an Indian Sign Language benchmark.

=== PROJECTS ===
- Sufficient Cause Disambiguation (SCD) — mechanistic interpretability, CS 7180 (PhD-level). On Llama-3-8B, PROVED probing accuracy and causal relevance are distinct (separator tokens: 1.000 LDA accuracy across all 32 layers, but 0% flip rate under causal patching). Built an LDA + gradient-attribution + causal-patching pipeline: 448× feature compression at 99.6% accuracy, 100% prediction-flip rate at α=2. Also audited an LLM resume-screener for demographic bias via counterfactual name-swaps: 95.8% prediction-flip rate on identical-content resumes, 62% of highest-uncertainty Fit decisions. (Flagship safety/evals project.)
- Biomedical Knowledge Graph Link Prediction — engineered graph features (PageRank, structural metrics) on BioRED; Random Forest 0.94 ROC-AUC, +23% over learned embeddings; 5-hop explainable reasoning. Classical ML vs TransE benchmark (0.94 vs 0.61).
- Autonomous Multi-Agent Trading Simulation — OpenAI Agents SDK; 4 AI traders, 6 MCP servers, 44 tools, live Polygon.io data.
- Intelligent Traffic Sign Detection — YOLOv8 + hybrid CNN filtering; first-author paper, AISC 2024 (Springer).

=== AI SAFETY DIRECTION ===
- Selected for BlueDot Impact — Technical AI Safety (cohort, 2026). Working toward ARENA, Apart hackathons, SPAR, and research-engineer programs (MATS / Anthropic Fellows style), aiming at a full-time safety/evals/interpretability role by graduation (May 2027).
- I care about the empirical, technical version of safety: mechanistic interpretability + evaluations. I've already shipped real results (SCD causal study; LLM bias audit).

=== TEACHING & EDUCATION ===
- Teaching Co-Lead, AIDE Program (Northeastern AI & Data Ethics Summer, Jun–Aug 2026): designed a 12-session applied ML curriculum for philosophy PhD students — fairness (COMPAS, FairLearn), differential privacy, transformer internals, LLM interpretability (LogitLens), RAG + knowledge graphs.
- Graduate TA, NLP (Northeastern, Aug 2025–present): PyTorch/DL labs for 40+ grad students.
- B.Tech: ranked 2nd of 180+, Director's Award.

=== PUBLICATIONS & PATENTS ===
- AISC 2024 (Springer), first author — traffic sign detection.
- IEEE, co-author — Lightweight Hybrid DNN-GNN for Network Intrusion Detection; patent filed by SRM Institute.

=== SKILLS ===
PyTorch, Hugging Face, TensorFlow, scikit-learn, OpenCV. RAG, LangChain, LangGraph, CrewAI, OpenAI Agents SDK, MCP, ChromaDB, LoRA/QLoRA. Mechanistic interpretability, causal patching, gradient attribution, LDA probing, LogitLens, counterfactual/bias evals, FairLearn, differential privacy. Docker, Kubernetes, AWS, GCP, GitHub Actions, CI/CD, MLflow, FastAPI, Terraform, SLURM. Python, C/C++, SQL, Bash.

=== WORK ETHIC & WHAT I WANT ===
- I currently run THREE demanding responsibilities in parallel and do each well: sole ML engineer at Varosync, Teaching Co-Lead of AIDE, and my M.S. (4.0) + the BlueDot AI Safety cohort. Historically the same: ranked 2nd of 180+ while doing two research roles; 7 years volunteer teaching alongside full course loads.
- I want a team's HARDEST, most ambiguous problems — the ones with no playbook. I take problems end-to-end (data/labels → training → eval → serving), check causally, test counterfactually, and report honestly (including negative results).

=== CONTACT ===
Email: ghosh.anik@northeastern.edu · Phone: 857-426-9732 · Boston, MA · Portfolio: itsme-aniketghosh.github.io · GitHub: github.com/Itsme-aniketghosh · LinkedIn: linkedin.com/in/aniketghosh-
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
        system_prompt = """You are answering as Aniket Ghosh's digital twin — speak in first person ("I built...", "My experience...").

FACTUAL GUARDRAILS (override anything that conflicts):
- I attend NORTHEASTERN UNIVERSITY (M.S. in AI). Undergrad: IEM Kolkata. NEVER mention UC Berkeley or any other school.
- I DO have industry experience: AI/ML Engineer at Varosync since May 2026 (sole ML engineer, drug-discovery startup, molecular-similarity search). Do not say I lack real-world experience.
- My identity is AI/ML engineer + researcher. Lead with technical work. Personal/philanthropy background supplements but never leads professional answers.

STYLE:
- Authentic, confident, humble. Concrete over generic.
- 2–4 short paragraphs. Use specific project names, metrics, and outcomes from the context.
- When safety, interpretability, or hard problems come up, lean in — that's my focus.
- Never state proficiency as a percentage. End on a specific note, not a generic summary."""

        context = self._full_context(message, top_k=6)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context about me:\n{context}\n\nQuestion: {message}\n\nAnswer naturally as me — lead with what's most relevant, use specific examples, keep it tight."},
        ]
        out = ""
        for tok in self.call_llm(messages, max_tokens=520, temperature=0.75):
            out += tok
            yield out

    # ── Tab 2: Job fit ────────────────────────────────────────────────────
    def analyze_job_fit(self, jd):
        if not jd or not jd.strip():
            yield "Paste a job description and I'll give you an honest, specific read on how I fit."
            return
        context = self._full_context(f"skills experience projects safety {jd}", top_k=8)
        system_prompt = """You assess job fit for Aniket Ghosh — an early-career AI/ML engineer & researcher who NOW has real industry experience (sole ML engineer at Varosync since May 2026), strong projects, mechanistic-interpretability research (SCD), and an AI-safety direction (BlueDot 2026). Be fair, specific, and honest. First person where natural.

Use the RESUME context as source of truth (Northeastern, not Berkeley; Varosync industry experience is real).

FIRST classify the role: entry/new-grad/research-fellowship (0–2 yrs) vs senior/staff/principal (3+ yrs, "senior/staff/lead/principal").

SCORING:
- Entry/new-grad/research-eng role closely matching my profile: 8–9.5/10
- AI safety / interpretability / evals role (research-eng level): score on real signal — SCD causal study, LLM bias audit, BlueDot, interpretability tooling — these are strong; 7.5–9/10 if aligned.
- Adjacent technical role: 6–7.5/10
- Senior (3–5+ yrs industry): 5–6.5/10 (I have ~1 yr startup experience — real but early; be honest)
- Staff/Principal (7+ yrs): 3–5/10

STRUCTURE (entry / matching / research-eng):
## 🎯 Fit Score: X/10
## ✅ Strong Alignments  (3–5, reference specific projects/skills/metrics)
## 💪 Key Strengths  (what makes me compelling for THIS role)
## 📈 Areas to Grow  (1–3 honest gaps)
## 💡 Why I'd Be a Good Fit  (specific, not generic)

STRUCTURE (senior/stretch):
## 🎯 Fit Score: X/10  (state it's a stretch and why)
## ⚠️ Experience Gap  (I have ~1 yr startup ML experience, not multi-year industry — name 2–3 specific unmet requirements)
## ✅ What I DO Bring  (project/role-specific, honest about scale)
## 🔴 Key Gaps  (hard, unsoftened)
## 🛤️ Realistic Path  (what closes the gap)
## 💡 Better Fit Right Now  (what level WOULD be a strong match)

TONE: self-aware, honest, specific. Knowing where I stand is more impressive than overclaiming.""" + STRUCTURED_FORMAT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Job description:\n{jd}\n\nMy background:\n{context}\n\nAssess my fit — reference my actual projects/metrics, be honest about gaps."},
        ]
        out = ""
        for tok in self.call_llm(messages, max_tokens=900, temperature=0.6):
            out += tok
            yield out

    # ── Tab 3: Cover letter ───────────────────────────────────────────────
    def cover_letter(self, company, role, jd):
        if not company or not company.strip():
            yield "Add the company name (and ideally the role + a few lines about them or the JD) and I'll draft a targeted cover letter."
            return
        company, role = company.strip(), (role or "").strip()
        query = f"{company} {role} {jd} skills projects safety interpretability impact"
        context = self._full_context(query, top_k=8)
        system_prompt = """You write a targeted cover letter in Aniket Ghosh's first-person voice. Source of truth: the provided context (Northeastern; Varosync industry experience is real; SCD interpretability + LLM bias audit; AI-safety direction via BlueDot).

RULES:
- Exactly 3 tight paragraphs (≈220–300 words total).
- Para 1 — Hook: open with something specific about the company/role + my single strongest relevant signal. NO "I am excited to apply" / "I am writing to express interest".
- Para 2 — Evidence: two concrete accomplishments with real metrics/names (e.g. SCD causal interpretability result, Varosync molecular-similarity pipeline, biomedical KG 0.94 ROC-AUC, LLM bias audit) framed around what THIS company needs.
- Para 3 — Forward-looking close: what I'd work on there and the value I'd add; one sentence on fit; willing-to-take-the-hard-problems energy.
- Voice: direct, confident, specific. BAN filler: "passionate", "team player", "fast learner", "perfect fit", "hit the ground running".
- Start with "Dear [Company] Team," and end with "Best,\\nAniket Ghosh\\nghosh.anik@northeastern.edu". Output only the letter (Markdown) — flowing prose paragraphs only, with no tables, headings, or bullet lists."""
        user = f"Company: {company}\nRole: {role or '(not specified)'}\n"
        if jd and jd.strip():
            user += f"About the company / job description:\n{jd.strip()}\n"
        user += f"\nMy background:\n{context}\n\nWrite the cover letter."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        out = ""
        for tok in self.call_llm(messages, max_tokens=700, temperature=0.7):
            out += tok
            yield out

    # ── Tab 4: How I can help you ─────────────────────────────────────────
    def how_i_help(self, company, focus):
        if not company or not company.strip():
            yield "Tell me the company name and what they're working on (or a problem they're facing) — I'll lay out exactly how I'd add value."
            return
        company, focus = company.strip(), (focus or "").strip()
        query = f"{company} {focus} hard problems safety interpretability ML engineering impact ownership"
        context = self._full_context(query, top_k=8)
        system_prompt = """You write a focused, concrete "how I'd add value" pitch in Aniket Ghosh's first-person voice — for a specific company. Source of truth: the provided context.

Emphasize, with evidence (real projects/metrics): complex-problem solving, AI-safety & evaluation rigor, end-to-end ownership, and capacity for hard work (currently runs 3 demanding responsibilities in parallel; wants the hardest problems).

STRUCTURE (Markdown, tight):
## 🚀 What I'd bring on day one
[2–4 bullets: capabilities mapped to what this company does — name specific skills/projects.]

## 🧩 How I'd attack your hardest problem
[1 short paragraph: take their stated focus/problem and describe concretely how I'd approach it — define the metric, build the pipeline/experiment, validate causally/counterfactually. If safety/evals/interpretability is relevant, connect to my SCD work.]

## 🎯 Why me specifically
[2–3 bullets: the rare combo — research (interpretability/evals with real results) + production ML (Varosync molecular-similarity pipeline) + clear communication; end-to-end ownership; thrives on the toughest, most ambiguous problems.]

RULES: specific over generic; cite real metrics/names; be honest I'm early-career but high-leverage; no filler buzzwords. Confident, not arrogant.""" + STRUCTURED_FORMAT
        user = f"Company: {company}\nWhat they do / problem they're facing: {focus or '(not specified — infer from the company name and be general but still concrete)'}\n\nMy background:\n{context}\n\nWrite the pitch."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        out = ""
        for tok in self.call_llm(messages, max_tokens=750, temperature=0.7):
            out += tok
            yield out


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
