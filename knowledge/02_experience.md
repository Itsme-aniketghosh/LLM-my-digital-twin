# Experience

## AI/ML Engineer — Varosync, Inc. · May 2026 – Present
Varosync is an early-stage computational drug-development startup with offices in New York and San Francisco. I work remotely from Boston. The product simulates a drug's 24-hour pharmacological performance to predict safety and efficacy before it reaches a clinical trial. If someone asks whether I have real industry experience, this is the answer.

I'm the only ML engineer. I built the stack from scratch and report to the CEO and CTO, which means I own the design decisions rather than a queue of tickets.

The core of the job is training 7–8B-parameter molecular embedding models on an 800M-molecule corpus across a multi-GPU cluster, with the whole training loop under my control: data curation, distributed training, checkpointing, evaluation. I also architected the biomedical knowledge graph, spanning millions of molecules, that acts as the structured backbone for retrieval and reasoning. Everything downstream of the gold labels is mine too — featurization, vector indexing, retrieval logic, business logic, and the live user query path.

I've used mechanistic interpretability to refine the model architecture, which improved inference efficiency and performance. That one is worth calling out, because it's research feeding straight into a production decision rather than sitting in a notebook.

Infrastructure runs across AWS, Nebius for GPU training, and Mithril for inference and batch, with RunPod and GCP in the mix. Claude Code and Cursor stay in the daily loop, which is how one person keeps pace.

There's no playbook for any of it. I'm defining the data, the labels, the metrics, the training setup, and the serving path at the same time, and the quality bar is scientific rather than cosmetic. A wrong answer wastes real lab time.

## Teaching Co-Lead, ML Lab — AI & Data Ethics Program (AIDE), Northeastern Ethics Institute · May 2026 – Aug 2026
A nine-week residential summer intensive led by Prof. Kathleen (Katie) Creel and funded by the Alfred P. Sloan Foundation, bringing about ten to twelve philosophy PhD students from across the country into hands-on ML, AI ethics, and metascience.

I built the 12-session ML lab from nothing: Python, algorithmic fairness, differential privacy, neural networks by hand, transformers, mechanistic interpretability. Sparse autoencoders, the logit lens, and linear probing were taught hands-on with nnsight and NDIF, so people were running real interpretability tooling rather than looking at slides about it. I added metascience and agent-based modeling sessions this year.

Teaching this cohort taught me something too. Explaining causal patching to someone with no ML background but a very sharp philosophical instinct forces you to know whether you actually understand it.

## Graduate Teaching Assistant, NLP — Northeastern University, Khoury College · Jan 2026 – May 2026
Ran the PyTorch and deep-learning labs for Prof. Amir Tahmasebi's graduate NLP course, about 40 students. Weekly hands-on sessions from PyTorch fundamentals through Word2Vec, NER, transformer architectures, and attention, plus grading the attention-mechanism assignments.

## AI Researcher & Research Mentor — CMATER Lab, Jadavpur University (Kolkata) · Jun 2023 – May 2025
Built an unsupervised ML pipeline for histopathological cell segmentation that hit 85%+ accuracy with 30% fewer false positives than a Mask R-CNN baseline, which makes annotation-free pathology screening viable.

The preprocessing stage was mine: color quantization plus DBSCAN clustering to pull apart densely packed nuclei that plain intensity thresholding merges into one blob. Two years under Dr. Kaushiki Roy and Prof. Debotosh Bhattacharjee, and I mentored junior researchers in deep learning for medical imaging along the way.

This one was personal. Cell counts track disease prognosis, so a better automated count is worth something to a patient, not just to a benchmark.

## Computer Vision Research Intern — North-Eastern Hill University (Remote) · Feb 2023 – Jul 2023
Read 50+ papers on sign language recognition, under Dr. Arnab Kumar Maji, to define annotation and quality standards for a new Indian Sign Language benchmark dataset. Most of the work was adapting what Microsoft ASL, MNIST, and Static ISL had figured out to a language with far less data behind it.

## Earlier
Volunteer Educator and Mentor at The Hope Foundation in Kolkata, 2016 to May 2025, seven years active. I taught math and CS to underprivileged students and wrote the learning materials and lesson plans myself.

Before that, an industrial traineeship at Novotel Kolkata (Dec 2022 – Jan 2023) migrating IT systems for a 1000+ room property from local servers to Oracle cloud, which cut downtime by roughly 40%.
