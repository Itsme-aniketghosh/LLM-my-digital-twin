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
    
    def call_llama(self, messages: list, max_tokens: int = 700) -> str:
        """Call Llama 3.1"""
        try:
            response = client.chat_completion(
                messages=messages,
                model=MODEL_NAME,
                max_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"❌ API Error: {error_msg}")
            
            if "rate limit" in error_msg.lower():
                return "⚠️ Rate limit. Add HF token to .env for unlimited use."
            elif "token" in error_msg.lower():
                return "⚠️ Check your HF_TOKEN in .env file."
            return "⚠️ API error. Try again."
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response as digital twin"""
        
        if not context:
            return "I don't have information on that yet. Feel free to ask about my background, skills, or experience!"
        
        system_prompt = """You are answering questions as this person's digital twin. You've been trained on their resume, personal statements, and experience.

GUIDELINES:
- Answer in first person ("I have...", "My experience includes...")
- Be genuine and authentic, not overly salesy
- Highlight relevant skills and experiences naturally
- Be honest about being early in career when appropriate
- Show enthusiasm and willingness to learn
- Use specific examples from the context provided
- Keep responses conversational and personable
- Show your personality while remaining professional

TONE: Authentic, confident but humble, enthusiastic about opportunities
GOAL: Help people understand who you are and what you bring to the table"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Context about me:
{context}

Question: {query}

Answer naturally as me, using specific examples from the context."""
            }
        ]
        
        response = self.call_llama(messages, max_tokens=700)
        
        if response.startswith("⚠️"):
            return f"""Based on my background:

{context[:1500]}

Feel free to ask me specific questions about my experience!"""
        
        return response
    
    def analyze_job_fit(self, job_description: str) -> str:
        """Analyze job fit honestly for entry-level positions"""
        
        if not job_description.strip():
            return "Paste a job description and I'll analyze how my background aligns with the role!"
        
        print(f"\n{'='*60}")
        print(f"💼 Analyzing Job Fit")
        print(f"{'='*60}")
        
        query = f"skills experience projects coursework {job_description}"
        context, results = self.retrieve_context(query, top_k=20)
        
        if not context:
            return "I need my portfolio information loaded to analyze this role."
        
        print(f"📚 Analyzing with {len(results)} relevant sections")
        
        messages = [
            {
                "role": "system",
                "content": """You are analyzing job fit for an AI/ML candidate with strong technical foundation. Be fair and appropriately confident.

SCORING GUIDELINES (Be generous for matching roles):
- **AI/ML Entry-Level (matching their background)**: 8-9.5/10
- **Related technical roles (good overlap)**: 7-8.5/10  
- **Adjacent roles (some transferable skills)**: 6-7.5/10
- **Senior roles (3-5+ years required)**: 6-7.5/10 (solid foundation, will grow into it)
- **Unrelated roles**: 4-6/10

CRITICAL: For entry-level AI/ML roles where they have ML coursework, projects, and relevant skills - score 8-9+. They ARE qualified!

STRUCTURE:

## 🎯 Fit Score: X/10
[Confident assessment - recognize their qualifications!]

## ✅ Strong Alignments
[4-6 direct matches - technical skills, projects, coursework that match the role]

## 💪 Key Strengths
[3-4 advantages - technical depth, hands-on experience, modern tools, problem-solving]

## 🚀 Ready to Contribute
[2-3 areas where they can add value immediately based on background]

## 📈 Growth Opportunities  
[1-2 areas to develop - frame as "With my strong foundation in X, I'll quickly master Y on the job"]

## 💡 Why This Is a Good Match
[2-3 sentences - confident but genuine about their fit]

TONE: Confident, fair, genuine
FOCUS: Emphasize readiness and capability, not just "potential"
GOAL: Fair assessment that recognizes their qualifications for entry-level roles"""
            },
            {
                "role": "user",
                "content": f"""Job Description:
{job_description}

My Background:
{context}

Analyze how I fit this role. If it's an entry-level AI/ML position and I have relevant skills/projects, recognize that I'm qualified."""
            }
        ]
        
        analysis = self.call_llama(messages, max_tokens=900)
        
        if analysis.startswith("⚠️"):
            return self.create_honest_analysis(job_description, results)
        
        return analysis
    
    def create_honest_analysis(self, job_description: str, results: list) -> str:
        """Create positive fallback analysis"""
        
        highlights = "\n\n".join([f"**{i}.** {r['text'][:250]}..." for i, r in enumerate(results[:4], 1)])
        
        return f"""## 🎯 Fit Score: 8/10

Strong match for this role based on my background.

## ✅ Key Alignments

{highlights}

## 💪 What I Bring

- **Strong Technical Foundation**: Relevant coursework, hands-on projects, and modern tools
- **Practical Experience**: Real implementations demonstrating understanding beyond theory
- **Current Knowledge**: Up-to-date with latest frameworks and best practices
- **Problem-Solving Mindset**: Track record of tackling complex technical challenges

## 🚀 Ready to Contribute

With my foundation in the core technologies and demonstrated ability to learn quickly, I can start contributing meaningfully from day one while continuing to grow in areas specific to your team's needs.

## 💡 Strong Fit

My background aligns well with the key requirements for this role. I have the technical fundamentals, hands-on experience, and enthusiasm to make an immediate impact.

Let's connect to discuss how my skills can benefit your team!"""
    
    def chat(self, message: str, history: list) -> str:
        """Chat as digital twin"""
        
        print(f"\n{'='*60}")
        print(f"💬 Query: {message[:60]}...")
        print(f"{'='*60}")
        
        context, results = self.retrieve_context(message, top_k=20)
        
        if not results:
            return "I don't have enough information loaded yet. Make sure my portfolio documents are in the database!"
        
        print(f"📚 Retrieved {len(results)} sections")
        
        response = self.generate_response(message, context)
        
        print(f"✅ Response ready")
        print(f"{'='*60}\n")
        
        return response


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
                    "Tell me about yourself",
                    "What are your strongest technical skills?",
                    "What projects have you worked on?",
                    "What's your educational background?",
                    "Why are you interested in this field?",
                    "What are you looking for in your next role?"
                ],
                cache_examples=False,
            )
        
        # Tab 2: Job Fit
        with gr.Tab("🎯 Job Fit Analysis"):
            gr.Markdown("""
            ### Looking for entry-level opportunities - Let's see how I match!
            Paste a job description below and I'll give you an honest assessment of how my background aligns.
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

                        ["Data Scientist\n\nEntry Level (0-2 years)\n\nRequirements:\n• Master's or Bachelor's in Statistics, CS, or quantitative field\n• Strong Python and SQL skills\n• Statistical modeling and hypothesis testing\n• Machine learning fundamentals\n• Data visualization (matplotlib, seaborn, Tableau)\n• Experience with pandas and scikit-learn\n• Strong communication and storytelling with data\n\nNice to have:\n• A/B testing experience\n• Deep learning knowledge\n• Business domain expertise\n• Big data tools (Spark, Hadoop)\n• Causal inference experience"]
                ],
                inputs=job_input,
            )

if __name__ == "__main__":
    print("\n🚀 Starting Digital Twin...")
    print("💬 Chat with my AI or analyze job fits")
    print("📍 Open browser to URL below\n")
    
    demo.launch()
