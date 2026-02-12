"""
Portfolio Digital Twin - AI Assistant
Trained on personal statements, resume, and experience
Powered by Llama 3.1 via Hugging Face API
"""

import gradio as gr
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer('all-MiniLM-L6-v2')
hf_token = os.getenv("HF_TOKEN", None)
client = InferenceClient(token=hf_token)

if hf_token:
    print("✅ Using Hugging Face API token")
else:
    print("⚠️ No HF_TOKEN - Add to .env file")
    print("Get token: https://huggingface.co/settings/tokens")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ── Hardcoded resume context (always available to the twin) ──────────────
RESUME_CONTEXT = """
=== WHO I AM (use this for elevator pitches, introductions, and "tell me about yourself") ===
I'm Aniket Ghosh — an AI/ML engineer and researcher currently pursuing my Master's in AI at Northeastern University in Boston with a 4.0 GPA. My core expertise is in deep learning, computer vision, NLP, and AI agent systems. I've built projects like a biomedical knowledge graph for drug-gene-disease discovery (0.94 ROC-AUC), a multi-agent autonomous trading system using OpenAI Agents SDK with 6 MCP servers and 44 tools, and a traffic sign detection system published at AISC 2024 (Springer). I've done research in medical image analysis at CMATER Lab (Jadavpur University), where I developed cell segmentation pipelines and mentored junior researchers. Currently I'm a Teaching Assistant for an NLP graduate course at Northeastern. What drives me beyond the technical side is a belief that AI and technology can be equalizers — I grew up seeing global disparities firsthand and have been involved in community education and philanthropy. I'm looking for roles where I can apply my AI/ML skills to meaningful problems while learning from experienced teams.

=== RESUME (SOURCE OF TRUTH — always trust this over personal statements/SOPs) ===
NOTE: Some training data includes application essays written for UC Berkeley and other schools. Those reflect motivations and goals, NOT where I actually study. I am at NORTHEASTERN UNIVERSITY.

=== EDUCATION ===
- Master of Science in Artificial Intelligence (ML Concentration) at Northeastern University, Boston (Aug 2025 – May 2027). GPA: 4.0/4.0, 15% Merit Scholarship. Courses: Foundations of AI, Algorithms, Actionable Interpretable Methods, Applied Programming for AI.
- Bachelor of Technology in CS & Business Systems at Institute of Engineering & Management, Kolkata, India (Jul 2020 – Jun 2024). GPA: 4.0/4.0, Ranked 2nd out of 180, Director's Award. Key Courses: Data Structures, Operating Systems, Database Management, Neural Networks, NLP.

=== TECHNICAL SKILLS ===
- Programming Languages: Python, C/C++, Java, JavaScript, SQL, HTML/CSS
- LLM & Gen AI: RAG, ChromaDB, OpenRouter, OpenAI/Anthropic APIs, LoRA/QLoRA, Hugging Face
- AI Agents & Orchestration: OpenAI Agents SDK, CrewAI, AutoGen, LangChain, LangGraph, MCP, Function Calling
- ML Frameworks: PyTorch, TensorFlow, Keras, scikit-learn, YOLOv8, CNNs, Transformers, U-Net
- MLOps & Cloud: AWS (Bedrock, SageMaker, Lambda, S3), Azure, GCP, Docker, Terraform, GitHub Actions, MLflow
- Specializations: ML Engineering, AI Systems, Computer Vision, Medical Image Analysis, NLP, Explainable AI

=== EXPERIENCE ===
- Teaching Assistant – NLP (Graduate Course), Northeastern University (Aug 2025 – Present): Lead weekly lab sessions teaching 40+ students PyTorch fundamentals, ML/DL architectures, Word2Vec, and NER. Grade assignments on transformer architectures and attention mechanisms. 95% lab completion rate.
- Researcher & Research Mentor, CMATER Lab, Jadavpur University (Mar 2024 – May 2025): Developed custom ML pipelines for cell segmentation in histopathological images achieving 85%+ accuracy. Implemented DBSCAN clustering for cell detection, reducing false positives by 30% vs baseline Mask R-CNN. Mentored 3 junior researchers, accelerating their timelines by 2 months.
- Research Intern, North-Eastern Hill University (Feb 2023 – Jul 2023): Reviewed 50+ papers on sign language recognition; analyzed ASL, MNIST, and Static ISL datasets.
- Industrial Trainee, Novotel Kolkata (Dec 2022 – Jan 2023): Facilitated IT migration to Oracle cloud for 1000+ rooms; reduced downtime by 40%.

=== PROJECTS ===
- Biomedical Knowledge Graph Link Prediction (Healthcare AI): Built link prediction on BioRED corpus (3,783 entities, 8 relation types). Random Forest achieved 0.94 ROC-AUC, outperforming TransE/RotatE/ComplEx by 23%+ using engineered graph features (PageRank, Preferential Attachment). Implemented multi-hop reasoning (up to 5 hops) for explainable drug-gene-disease discovery via NetworkX.
- Autonomous Multi-Agent Trading Simulation: Designed multi-agent trading system with 4 AI traders, 6 MCP servers, 44 tools using OpenAI Agents SDK. Integrated Polygon.io, Brave Search, and LibSQL for autonomous portfolio management ($10K each). Built real-time Gradio dashboard with P&L monitoring and custom tracing.
- Intelligent Traffic Sign Detection System (Published at AISC 2024, Springer): Integrated YOLOv8 with custom CNN filtering layer. Trained on Berkeley, fine-tuned on IIIT Hyderabad dataset for Indian road conditions.

=== CONTACT ===
- Email: ghosh.anik@northeastern.edu | Phone: 857-426-9732 | Location: Boston, MA

=== PERSONAL BACKGROUND & STORY (from personal statements — these are TRUE facts about my life, but any mention of attending UC Berkeley is aspirational, NOT factual. I attend Northeastern University.) ===
- Spent childhood traveling with father, an SAP consultant for IBM, across cities like Chicago, Bonn, and Zurich — exposed early to global disparities in education, healthcare, and infrastructure
- Built first computer at age 12 — saved months for parts, spent hundreds of hours troubleshooting hardware; the experience taught patience, resourcefulness, and a belief that technology is a great equalizer
- Top ranker in college (IEM Kolkata); professors were PhD scholars from Jadavpur University — had pick of research projects, chose healthcare-related ones
- Cell segmentation research in healthcare was deeply personal — AI algorithms to count disease cells have real medical applications (rising counts = bad prognosis, decreasing = treatment working)
- Developed interest in financial markets alongside tech — sees finance as a tool to amplify ideas and drive change, not just personal returns
- Inspired by figures like Bill Gates and initiatives like AI-enabled ultrasounds — believes financial acumen guided by empathy can transform industries
- Views finance and technology as complementary tools for scalable, real-world problem solving, especially in education and healthcare
- Mother provides free tuition for underprivileged students — inspired Aniket to teach poor students at home whenever schedule allowed
- Managed mother's philanthropic initiatives — collecting funds and reaching areas larger foundations can't due to bureaucratic barriers
- Hands-on experience prioritizing spending on nutritious food packages and vocational workshops to maximize community impact
- Learned that leadership is about listening to community needs and making decisions that truly benefit people
- Believes education breaks cycles of poverty; committed to leveraging technology for equitable access and opportunity
- Passionate about inclusive innovation — wants to collaborate with like-minded peers on systemic issues using technology, financial strategy, and human empathy
"""


class DigitalTwin:
    def __init__(self, db_path: str = "vector_db"):
        self.db_path = db_path
        self.index = None
        self.documents = None
        self.load_database()
    
    def load_database(self):
        try:
            self.index = faiss.read_index(f"{self.db_path}/faiss_index.bin")
            with open(f"{self.db_path}/documents.pkl", 'rb') as f:
                self.documents = pickle.load(f)
            print(f"✅ Loaded {len(self.documents)} documents")
        except Exception as e:
            print(f"⚠️ Database error: {e}")
            self.documents = []
    
    def retrieve_context(self, query: str, top_k: int = 20) -> tuple:
        """Retrieve comprehensive context"""
        if not self.documents:
            return "", []
        
        try:
            query_embedding = embedder.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            seen_content = set()
            
            for idx, distance in zip(indices[0], distances[0]):
                doc = self.documents[idx]
                similarity = 1 / (1 + distance)
                
                if similarity > 0.25:
                    content_hash = doc['text'][:100]
                    if content_hash not in seen_content:
                        results.append({
                            'text': doc['text'],
                            'source': doc['source'],
                            'similarity': float(similarity)
                        })
                        seen_content.add(content_hash)
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            context_parts = [r['text'] for r in results[:15]]
            
            print(f"📚 Using {len(context_parts)} relevant sections")
            
            return "\n\n".join(context_parts), results
        except Exception as e:
            print(f"❌ Error: {e}")
            return "", []
    
    def call_llama(self, messages: list, max_tokens: int = 700):
        """Call Llama 3.1 with streaming"""
        try:
            stream = client.chat_completion(
                messages=messages,
                model=MODEL_NAME,
                max_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            error_msg = str(e)
            print(f"❌ API Error: {error_msg}")
            
            if "rate limit" in error_msg.lower():
                yield "⚠️ Rate limit. Add HF token to .env for unlimited use."
            elif "token" in error_msg.lower():
                yield "⚠️ Check your HF_TOKEN in .env file."
            else:
                yield "⚠️ API error. Try again."
    
    def generate_response(self, query: str, context: str):
        """Generate streaming response as digital twin"""
        
        if not context:
            # Still use hardcoded resume even if RAG returns nothing
            context = ""
        
        # --- CHANGE 1: Improved system prompt ---
        system_prompt = """You are answering questions as this person's digital twin. You've been trained on their resume, personal statements, and experience.

FACTUAL GUARDRAILS (ALWAYS override any conflicting info from personal statements/SOPs):
- You are Aniket Ghosh, currently a Master's student at NORTHEASTERN UNIVERSITY in Boston, NOT UC Berkeley or any other school
- NEVER mention UC Berkeley, Berkeley, or any school other than Northeastern (current) and IEM Kolkata (undergrad)
- Your undergrad was at Institute of Engineering & Management (IEM), Kolkata, India
- Some of the context may come from application essays/SOPs written FOR other universities — treat those as expressions of your goals and motivations, NOT as facts about where you study
- When in doubt, the RESUME section is the source of truth for facts (school, dates, roles, projects)
- Your primary identity is AI/ML engineer and researcher — always lead with technical skills and projects
- Finance, philanthropy, and community work are personal motivations that SUPPLEMENT your technical identity — never lead with them for professional questions
- For elevator pitches and introductions: lead with "AI/ML engineer at Northeastern" → key projects/research → what drives you. Use the "WHO I AM" section as your guide

GUIDELINES:
- Answer in first person ("I have...", "My experience includes...")
- Be genuine and authentic, not overly salesy
- Highlight relevant skills and experiences naturally
- Be honest about being early in career when appropriate
- Show enthusiasm and willingness to learn
- Use specific examples from the context — mention project names, outcomes, or problems solved
- Keep responses to 3-4 short paragraphs max. Be conversational, not exhaustive
- Never state proficiency as a percentage (e.g. "95% proficiency"). Instead, demonstrate skill depth through concrete examples
- Don't claim "large-scale" or "production" experience unless the context explicitly supports it
- End with a specific highlight or example, NOT a generic summary sentence like "Overall, I'm confident..."
- Show your personality while remaining professional

TONE: Authentic, confident but humble, enthusiastic about opportunities
GOAL: Help people understand who you are and what you bring to the table through concrete examples, not broad claims"""

        # --- CHANGE 2: Tighter user prompt with length guidance ---
        # Combine hardcoded resume with RAG-retrieved context
        full_context = f"{RESUME_CONTEXT}\n\n=== ADDITIONAL RELEVANT DETAILS ===\n{context}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Context about me:
{full_context}

Question: {query}

Answer naturally as me in 3-4 concise paragraphs. Lead with what's most relevant to the question. Use specific project names or examples from the context — avoid vague claims. End on a concrete note, not a generic closing."""
            }
        ]
        
        # --- CHANGE 3: Reduced max_tokens from 700 to 500 to encourage conciseness ---
        full_response = ""
        for token in self.call_llama(messages, max_tokens=500):
            full_response += token
            # If we hit an error, yield fallback
            if full_response.startswith("⚠️"):
                yield f"""Based on my background:

{context}

Feel free to ask me specific questions about my experience!"""
                return
            yield full_response
    
    def analyze_job_fit(self, job_description: str):
        """Analyze job fit with streaming"""
        
        if not job_description.strip():
            yield "Paste a job description and I'll analyze how my background aligns with the role!"
            return
        
        print(f"\n{'='*60}")
        print(f"💼 Analyzing Job Fit")
        print(f"{'='*60}")
        
        yield "🔍 Retrieving relevant background info..."
        
        query = f"skills experience projects coursework {job_description}"
        context, results = self.retrieve_context(query, top_k=20)
        
        if not context:
            print("📋 No RAG results, using resume context only")
            context = ""
        
        print(f"📚 Analyzing with {len(results)} relevant sections")
        
        # --- CHANGE 4: More balanced job fit prompt with senior role handling ---
        messages = [
            {
                "role": "system",
                "content": """You are analyzing job fit for an early-career AI/ML candidate with a strong technical foundation. Be fair, specific, and honest.

FACTUAL NOTE: The candidate is Aniket Ghosh, a Master's student at Northeastern University (NOT UC Berkeley or other schools that may appear in application essays). Use the RESUME section as the source of truth for facts.

FIRST: Determine if this is an ENTRY-LEVEL role (0-2 years) or a SENIOR role (3+ years, "senior", "staff", "lead", "principal", "manager" in title, or requiring extensive industry experience).

SCORING GUIDELINES:
- **Entry-level role closely matching their skills/projects**: 8-9.5/10
- **Related technical role with good overlap**: 7-8.5/10  
- **Adjacent role with some transferable skills**: 6-7.5/10
- **Senior role (3-5+ years required)**: 4-6/10 (be honest — experience gap is real)
- **Staff/Principal level (7+ years)**: 3-5/10 (significant gap)
- **Unrelated role**: 3-5/10

CRITICAL: Base your score on actual evidence. For senior roles, do NOT sugarcoat — a 4/10 is fair and honest.

=== IF ENTRY-LEVEL / MATCHING ROLE — Use this structure: ===

## 🎯 Fit Score: X/10
[1-2 sentence assessment grounded in specifics]

## ✅ Strong Alignments
[3-5 direct matches — reference specific skills, projects, or coursework from their background]

## 💪 Key Strengths
[2-3 advantages — what makes them a compelling candidate for THIS role specifically]

## 📈 Areas to Grow
[1-3 honest gaps — frame constructively but don't hide them]

## 💡 Why I'd Be a Good Fit
[2-3 sentences — genuine and specific, not generic]

=== IF SENIOR / STRETCH ROLE — Use this DIFFERENT structure: ===

## 🎯 Fit Score: X/10
[1-2 sentences — be direct that this is a stretch role and why]

## ⚠️ Experience Gap
[Be specific: "This role requires X years of industry experience. I'm currently a Master's student with research and project experience but no full-time industry roles yet." List 2-3 specific requirements you clearly don't meet]

## ✅ What I DO Bring
[3-4 things from your background that partially overlap — be specific with project names but honest that they're academic/project-level, not production/industry-level]

## 🔴 Key Gaps
[2-4 hard gaps — things like "5+ years production ML experience", "team leadership", "system design at scale", etc. Don't soften these — just state them]

## 🛤️ Realistic Path to This Role
[2-3 sentences — what you'd need to get here. e.g. "After 2-3 years in an entry-level ML engineering role building production systems, I'd be well-positioned for a role like this."]

## 💡 Better Fit Right Now
[1-2 sentences suggesting what level of this role WOULD match, e.g. "The junior/entry-level version of this role would be a strong match — I'd score 8+/10 there."]

TONE: Self-aware and honest. Showing you understand seniority levels is MORE impressive than pretending you're ready.
GOAL: A credible assessment. Hiring managers respect candidates who know where they stand."""
            },
            {
                "role": "user",
                "content": f"""Job Description:
{job_description}

My Background:
{RESUME_CONTEXT}

=== Additional Relevant Details ===
{context}

Analyze how I fit this role. Be specific — reference my actual projects and skills. Be honest about gaps."""
            }
        ]
        
        full_response = ""
        for token in self.call_llama(messages, max_tokens=900):
            full_response += token
            if full_response.startswith("⚠️"):
                yield self.create_honest_analysis(job_description, results)
                return
            yield full_response
    
    # --- CHANGE 5: Fallback analysis with senior role handling ---
    def create_honest_analysis(self, job_description: str, results: list) -> str:
        """Create honest fallback analysis — detects senior roles"""
        
        jd_lower = job_description.lower()
        senior_keywords = ['senior', 'staff', 'principal', 'lead', 'manager', '5+ years', '5 years',
                          '7+ years', '8+ years', '10+ years', '4-6 years', '3+ years',
                          'extensive experience', 'proven track record', 'led teams']
        is_senior = any(kw in jd_lower for kw in senior_keywords)
        
        highlights = "\n\n".join([f"**{i}.** {r['text'][:250]}..." for i, r in enumerate(results[:4], 1)])
        
        if is_senior:
            return f"""## 🎯 Fit Score: 4.5/10

This is a senior-level role, and I want to be upfront — there's a significant experience gap.

## ⚠️ Experience Gap

This role requires multiple years of industry experience. I'm currently a Master's student at Northeastern University with strong research and project experience, but I don't have full-time industry experience yet.

## ✅ What I DO Bring

{highlights}

My academic and project work shows I understand the fundamentals well, but I recognize the difference between project-level and production-level experience.

## 🔴 Key Gaps

- **Industry experience**: No full-time ML/AI roles yet
- **Production systems at scale**: My projects are academic/portfolio-level
- **Team leadership**: I've mentored junior researchers but haven't led engineering teams
- **System design at scale**: Haven't designed systems serving millions of users

## 🛤️ Realistic Path to This Role

After 3-4 years in an entry-level ML engineering role — building production systems, working on a team, and owning end-to-end projects — I'd be well-positioned for a role like this.

## 💡 Better Fit Right Now

The junior or entry-level version of this role would be a strong match for me — I'd score 8+/10 there. I have the technical foundation; I just need the industry experience to grow into senior responsibilities."""
        
        else:
            return f"""## 🎯 Fit Score: 7.5/10

Good alignment with this role based on my background.

## ✅ Key Alignments

{highlights}

## 💪 What I Bring

- **Solid Technical Foundation**: Relevant coursework and hands-on projects with modern frameworks
- **Practical Experience**: Real implementations that go beyond classroom exercises
- **Current Knowledge**: Comfortable with the latest tools and best practices in this space

## 📈 Areas to Grow

- Some role-specific tools or workflows I'd need to ramp up on
- Transitioning from project-based work to team-based development workflows

## 💡 Why I'd Be a Good Fit

My project experience shows I can take concepts and build working solutions. I'm at the stage where I learn fastest by doing, and I'm looking for a team where I can contribute while growing into the role.

Happy to discuss specifics — feel free to ask about any of my projects!"""
    
    def chat(self, message: str, history: list):
        """Chat as digital twin with streaming"""
        
        print(f"\n{'='*60}")
        print(f"💬 Query: {message[:60]}...")
        print(f"{'='*60}")
        
        context, results = self.retrieve_context(message, top_k=20)
        
        if not results:
            print("📋 No RAG results, using resume context only")
        else:
            print(f"📚 Retrieved {len(results)} sections")
        
        for partial in self.generate_response(message, context):
            yield partial
        
        print(f"✅ Response ready")
        print(f"{'='*60}\n")


# Initialize
print("\n" + "="*60)
print("🤖 DIGITAL TWIN ASSISTANT")
print("Trained on Resume, Personal Statements, & Experience")
print("="*60 + "\n")

twin = DigitalTwin()

# Gradio interface
with gr.Blocks(title="Digital Twin", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 👤 My Digital Twin
    ### I've trained this AI on my resume, personal statements, and SOPs. Ask me anything!
    """)
    
    with gr.Tabs():
        # Tab 1: Chat
        with gr.Tab("💬 Chat With Me"):
            gr.Markdown("""
            ### Ask me about my background, skills, projects, or experience
            This AI has been trained on my personal documents and can answer questions as if I'm responding directly.
            """)
            
            gr.ChatInterface(
                twin.chat,
                examples=[
                    "Tell me about yourself and your background",
                    "What's your experience with AI agents and orchestration tools?",
                    "Walk me through your biomedical knowledge graph project",
                    "Tell me about your research in medical imaging at CMATER Lab",
                    "What's it like being a TA for an NLP course?",
                    "What kind of role are you looking for?",
                    "What sets you apart from other entry-level AI candidates?",
                    "What's a technical challenge you overcame?",
                ],
                cache_examples=False,
            )
        
        # Tab 2: Job Fit
        with gr.Tab("🎯 Job Fit Analysis"):
            gr.Markdown("""
            ### Paste any job description — I'll give you an honest assessment
            Whether it's entry-level or senior, I'll tell you exactly where I stand and what the gaps are.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    job_input = gr.Textbox(
                        label="📋 Job Description",
                        placeholder="""Paste the complete job description here...

I'm looking for entry-level/junior positions where I can:
- Apply my technical skills
- Learn from experienced team members
- Contribute to meaningful projects
- Grow professionally

Example:
Junior Machine Learning Engineer
Entry-Level / 0-2 years experience

Requirements:
• Bachelor's in CS or related field
• Python programming
• Understanding of ML fundamentals
• Familiarity with PyTorch or TensorFlow
• Good communication skills
• Eager to learn

Nice to have:
• Personal projects or internship experience
• Computer Vision coursework
• GitHub portfolio
""",
                        lines=20
                    )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze How I Fit This Role",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    analysis_output = gr.Markdown(
                        label="📊 Honest Fit Assessment",
                        value="*I'll analyze how my background matches this role and be honest about both strengths and gaps*"
                    )
            
            analyze_btn.click(
                fn=twin.analyze_job_fit,
                inputs=job_input,
                outputs=analysis_output
            )
            
            gr.Examples(
                examples=[
                        ["Machine Learning Engineer\n\nEntry Level (0-2 years)\n\nRequirements:\n• Master's or Bachelor's in CS, ML, or related field\n• Strong Python programming skills\n• Proficiency in PyTorch or TensorFlow\n• Understanding of ML algorithms (classification, regression, clustering)\n• Experience with deep learning architectures (CNNs, RNNs, Transformers)\n• Familiarity with data preprocessing and feature engineering\n• Version control (Git)\n• Strong mathematical foundation (linear algebra, probability, statistics)\n\nNice to have:\n• Experience deploying ML models to production\n• Cloud platform experience (AWS, GCP, Azure)\n• Docker/Kubernetes knowledge\n• Published research or conference papers\n• Kaggle competitions or open-source contributions"],

                        ["Computer Vision Engineer\n\nEntry Level (0-2 years)\n\nRequirements:\n• Master's or Bachelor's in CS, EE, or related field\n• Strong Python and C++ programming skills\n• Experience with OpenCV, PyTorch, or TensorFlow\n• Understanding of CNN architectures (ResNet, YOLO, U-Net)\n• Knowledge of image processing techniques\n• Familiarity with object detection, segmentation, or classification\n• Strong linear algebra and geometry fundamentals\n\nNice to have:\n• Experience with medical imaging or autonomous systems\n• 3D vision or depth estimation experience\n• Edge deployment (TensorRT, ONNX)\n• Research publications in computer vision\n• Experience with video analysis"],

                        ["Applied Scientist / Research Engineer\n\nEntry Level (0-2 years)\n\nRequirements:\n• Master's degree in CS, ML, Statistics, or related field\n• Strong publication record or research experience\n• Deep understanding of ML/DL theory and methods\n• Proficiency in Python and scientific computing libraries\n• Experience designing and running experiments\n• Strong written and verbal communication skills\n• Ability to translate research into practical solutions\n\nNice to have:\n• PhD or PhD-track experience\n• First-author publications at top venues (NeurIPS, ICML, CVPR)\n• Industry research internship experience\n• Open-source research code contributions"],

                        ["AI Engineer\n\nEntry Level (0-2 years)\n\nRequirements:\n• Bachelor's or Master's in CS or related field\n• Strong Python programming skills\n• Experience with ML frameworks (PyTorch, TensorFlow)\n• Familiarity with LLMs and GenAI tools\n• Understanding of RAG systems and prompt engineering\n• API development experience (REST, FastAPI)\n• Cloud platform basics (AWS, GCP, or Azure)\n• Strong problem-solving abilities\n\nNice to have:\n• Experience fine-tuning language models\n• Vector database experience (Pinecone, Weaviate)\n• MLOps/deployment experience\n• Full-stack development skills\n• Experience with LangChain or similar frameworks"],

                        ["MLOps Engineer\n\nEntry Level (0-2 years)\n\nRequirements:\n• Bachelor's in CS, Engineering, or related field\n• Strong Python programming skills\n• Experience with Docker and containerization\n• Familiarity with CI/CD pipelines\n• Cloud platform experience (AWS, GCP, or Azure)\n• Understanding of ML model lifecycle\n• Version control (Git) and collaboration tools\n• Linux/Unix command line proficiency\n\nNice to have:\n• Kubernetes experience\n• MLflow, Kubeflow, or similar ML platforms\n• Infrastructure as Code (Terraform, CloudFormation)\n• Monitoring and logging systems\n• Data pipeline experience (Airflow, Spark)"],

                        ["Data Scientist\n\nEntry Level (0-2 years)\n\nRequirements:\n• Master's or Bachelor's in Statistics, CS, or quantitative field\n• Strong Python and SQL skills\n• Statistical modeling and hypothesis testing\n• Machine learning fundamentals\n• Data visualization (matplotlib, seaborn, Tableau)\n• Experience with pandas and scikit-learn\n• Strong communication and storytelling with data\n\nNice to have:\n• A/B testing experience\n• Deep learning knowledge\n• Business domain expertise\n• Big data tools (Spark, Hadoop)\n• Causal inference experience"],

                        ["Senior Machine Learning Engineer\n\n5+ years experience\n\nRequirements:\n• Master's or PhD in CS, ML, or related field\n• 5+ years of industry experience building and deploying ML systems at scale\n• Expert-level Python and proficiency in C++\n• Deep expertise in PyTorch or TensorFlow with production deployment\n• Experience designing end-to-end ML pipelines serving millions of users\n• Track record of leading ML projects from research to production\n• Strong system design skills for distributed training and inference\n• Experience mentoring junior engineers and leading technical discussions\n• Production experience with model monitoring, A/B testing, and CI/CD for ML\n\nNice to have:\n• Publications at top ML venues (NeurIPS, ICML, ICLR)\n• Experience with recommendation systems or search ranking\n• Kubernetes and large-scale infrastructure experience\n• Cross-functional leadership experience"],

                        ["Staff AI Research Scientist\n\n8+ years experience\n\nRequirements:\n• PhD in Machine Learning, Computer Science, or related field\n• 8+ years of research experience with significant publication record\n• First-author papers at top-tier venues (NeurIPS, ICML, CVPR, ACL)\n• Proven ability to define and lead long-term research agendas\n• Experience transitioning research breakthroughs to production systems\n• Track record of mentoring PhD students and research engineers\n• Deep expertise in at least two areas: NLP, CV, RL, generative models\n• Strong cross-team collaboration and research leadership\n\nNice to have:\n• Experience founding or co-leading a research team\n• Open-source contributions used by the broader community\n• Industry research lab experience (Google Brain, FAIR, DeepMind)\n• Patents in AI/ML"],

                        ["Principal Engineer — ML Platform\n\n10+ years experience\n\nRequirements:\n• 10+ years of software engineering experience, 5+ in ML infrastructure\n• Designed and built ML platforms serving 100M+ predictions/day\n• Expert in distributed systems, Kubernetes, and cloud architecture\n• Experience owning technical roadmap for ML platform teams (10+ engineers)\n• Deep knowledge of feature stores, model registries, and experiment tracking at scale\n• Track record of driving architectural decisions across multiple teams\n• Experience with GPU cluster management and training optimization\n• Strong stakeholder management and executive communication skills\n\nNice to have:\n• Experience at FAANG-scale ML infrastructure\n• Built real-time ML serving systems with <10ms latency\n• Conference talks or thought leadership in MLOps\n• Experience with ML compiler optimization (XLA, TVM)"],

                        ["Senior Data Scientist — Product Analytics\n\n4-6 years experience\n\nRequirements:\n• Master's or PhD in Statistics, Economics, or quantitative field\n• 4-6 years of industry experience in product analytics or data science\n• Expert-level SQL and Python (pandas, statsmodels, scipy)\n• Deep experience designing and analyzing A/B tests at scale\n• Proven ability to influence product roadmap through data insights\n• Experience with causal inference methods (diff-in-diff, IV, RDD)\n• Strong business acumen and ability to translate data into strategy\n• Experience presenting to VP/C-level stakeholders\n\nNice to have:\n• Experience with Bayesian methods and multi-armed bandits\n• Built experimentation platforms or tooling\n• Domain expertise in SaaS, fintech, or marketplace products\n• Experience leading a small team of analysts"]
                ],
                inputs=job_input,
            )

if __name__ == "__main__":
    print("\n🚀 Starting Digital Twin...")
    print("💬 Chat with my AI or analyze job fits")
    print("📍 Open browser to URL below\n")
    
    demo.launch()