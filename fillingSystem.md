# Complete Project Structure: RL Portfolio Budgeting System + Publications

## Overview
This structure supports:
1. English Q1 journal article (LaTeX)
2. Persian M.Sc. thesis documentation (XePersian/LaTeX)
3. Full system implementation (RL agent, baselines, backend, frontend)
4. Reproducible experiments and data
5. Documentation and deployment guides

---

## Root Directory Structure

```
rl-portfolio-budgeting/
├── README.md                          # Project overview, quick start
├── LICENSE                            # Open-source license (MIT recommended)
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (alternative)
├── docker-compose.yml                 # Full system deployment
├── Makefile                           # Common commands (train, test, deploy)
│
├── paper/                             # English journal article
│   ├── main.tex                       # Main LaTeX file
│   ├── references.bib                 # Bibliography
│   ├── sections/
│   │   ├── 01-introduction.tex
│   │   ├── 02-literature.tex
│   │   ├── 03-problem-formulation.tex
│   │   ├── 04-methodology.tex
│   │   ├── 05-experiments.tex
│   │   ├── 06-system-architecture.tex
│   │   ├── 07-managerial-insights.tex
│   │   ├── 08-conclusion.tex
│   │   └── appendix.tex
│   ├── figures/                       # All paper figures
│   │   ├── architecture-diagram.pdf
│   │   ├── speedup-comparison.pdf
│   │   ├── quality-comparison.pdf
│   │   ├── scalability-plot.pdf
│   │   ├── sensitivity-analysis.pdf
│   │   ├── attention-heatmap.pdf
│   │   └── ui-screenshots/
│   ├── tables/                        # LaTeX table files
│   │   ├── baseline-comparison.tex
│   │   ├── scalability-results.tex
│   │   └── finetuning-results.tex
│   ├── supplementary/                 # Supplementary material
│   │   ├── supplementary.tex
│   │   ├── hyperparameters.tex
│   │   ├── additional-experiments.tex
│   │   └── code-snippets.tex
│   └── submission/                    # Journal-ready files
│       ├── manuscript.pdf
│       ├── cover-letter.tex
│       ├── response-to-reviewers.tex
│       └── highlights.txt
│
├── thesis/                            # Persian M.Sc. thesis
│   ├── main.tex                       # Main XePersian file
│   ├── references-fa.bib              # Persian bibliography
│   ├── references-en.bib              # English bibliography
│   ├── config/
│   │   ├── preamble.tex               # XePersian setup
│   │   ├── commands.tex               # Custom commands
│   │   └── university-style.sty       # University style file
│   ├── frontmatter/
│   │   ├── title-page.tex             # Persian title page
│   │   ├── abstract-fa.tex            # Persian abstract
│   │   ├── abstract-en.tex            # English abstract
│   │   ├── acknowledgments.tex        # تشکر و قدردانی
│   │   ├── dedication.tex             # تقدیم به
│   │   └── table-of-contents.tex
│   ├── chapters/
│   │   ├── 01-introduction.tex        # فصل اول: مقدمه
│   │   ├── 02-literature.tex          # فصل دوم: مروری بر ادبیات
│   │   ├── 03-methodology.tex         # فصل سوم: روش‌شناسی
│   │   ├── 04-implementation.tex      # فصل چهارم: پیاده‌سازی
│   │   ├── 05-experiments.tex         # فصل پنجم: آزمایش‌ها و نتایج
│   │   ├── 06-system.tex              # فصل ششم: سیستم نهایی
│   │   └── 07-conclusion.tex          # فصل هفتم: نتیجه‌گیری
│   ├── appendices/
│   │   ├── appendix-a-code.tex        # پیوست الف: کدها
│   │   ├── appendix-b-data.tex        # پیوست ب: داده‌ها
│   │   ├── appendix-c-results.tex     # پیوست ج: نتایج تکمیلی
│   │   └── appendix-d-manual.tex      # پیوست د: راهنمای استفاده
│   ├── figures/                       # Shared with paper/ via symlink
│   └── submission/
│       ├── thesis-final.pdf
│       └── defense-presentation.pdf
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── config.py                      # Global configuration
│   │
│   ├── environment/                   # Synthetic environment
│   │   ├── __init__.py
│   │   ├── generator.py               # Instance generation
│   │   ├── distributions.py           # Cost/return distributions
│   │   ├── calibration.py             # Literature calibration
│   │   ├── validator.py               # Statistical validation
│   │   └── portfolio.py               # Portfolio class
│   │
│   ├── agents/                        # RL agents
│   │   ├── __init__.py
│   │   ├── ppo_agent.py               # PPO implementation
│   │   ├── network.py                 # Policy/value networks
│   │   ├── attention.py               # Attention mechanism (optional)
│   │   ├── trainer.py                 # Training loop
│   │   └── finetuner.py               # Fine-tuning module
│   │
│   ├── baselines/                     # Baseline methods
│   │   ├── __init__.py
│   │   ├── greedy.py                  # Greedy heuristic
│   │   ├── mip_solver.py              # Gurobi MIP wrapper
│   │   ├── stochastic_prog.py         # Stochastic programming
│   │   └── perfect_info.py            # Perfect-information upper bound
│   │
│   ├── evaluation/                    # Evaluation framework
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Performance metrics
│   │   ├── comparator.py              # Baseline comparison
│   │   ├── sensitivity.py             # Sensitivity analysis
│   │   └── interpretability.py        # SHAP/attention analysis
│   │
│   ├── backend/                       # Backend API
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI application
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── allocate.py            # /allocate endpoint
│   │   │   ├── finetune.py            # /finetune endpoint
│   │   │   ├── evaluate.py            # /evaluate endpoint
│   │   │   └── health.py              # /health endpoint
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── request.py             # Request schemas
│   │   │   └── response.py            # Response schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py           # RL inference service
│   │   │   ├── finetuning.py          # Fine-tuning service
│   │   │   └── storage.py             # Model storage
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── validation.py          # Input validation
│   │       └── logging.py             # Logging setup
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── data_loader.py             # Data loading utilities
│       ├── visualization.py           # Plotting functions
│       ├── logger.py                  # Logging utilities
│       └── reproducibility.py         # Random seed management
│
├── frontend/                          # Frontend UI
│   ├── package.json                   # Node dependencies
│   ├── vite.config.js                 # Vite configuration
│   ├── index.html
│   ├── public/
│   │   ├── favicon.ico
│   │   └── logo.png
│   ├── src/
│   │   ├── main.js                    # Entry point
│   │   ├── App.vue                    # Root component
│   │   ├── router/
│   │   │   └── index.js               # Vue Router
│   │   ├── views/
│   │   │   ├── Home.vue               # Landing page
│   │   │   ├── Upload.vue             # Project upload
│   │   │   ├── Allocate.vue           # Allocation interface
│   │   │   ├── Results.vue            # Results visualization
│   │   │   ├── Finetune.vue           # Fine-tuning interface
│   │   │   └── Compare.vue            # Scenario comparison
│   │   ├── components/
│   │   │   ├── ProjectTable.vue       # Project list table
│   │   │   ├── AllocationChart.vue    # Allocation visualization
│   │   │   ├── MetricsCard.vue        # Performance metrics
│   │   │   └── FileUpload.vue         # File upload component
│   │   ├── services/
│   │   │   └── api.js                 # API client
│   │   ├── store/
│   │   │   └── index.js               # Vuex/Pinia store
│   │   └── assets/
│   │       ├── styles/
│   │       │   └── main.css
│   │       └── images/
│   └── dist/                          # Build output
│
├── experiments/                       # Experiment scripts
│   ├── 01_environment_validation.py   # Validate synthetic data
│   ├── 02_baseline_comparison.py      # Run all baselines
│   ├── 03_rl_training.py              # Train RL agent
│   ├── 04_scalability_analysis.py     # Scalability experiments
│   ├── 05_sensitivity_analysis.py     # Sensitivity experiments
│   ├── 06_finetuning_study.py         # Fine-tuning case study
│   ├── 07_interpretability.py         # SHAP/attention analysis
│   └── 08_generate_figures.py         # Generate all paper figures
│
├── data/                              # Data directory
│   ├── synthetic/                     # Synthetic instances
│   │   ├── parameters.json            # Calibrated parameters
│   │   ├── instances/
│   │   │   ├── train/                 # 8000 instances
│   │   │   ├── val/                   # 1000 instances
│   │   │   └── test/                  # 1000 instances
│   │   └── validation_stats.json      # Statistical validation
│   ├── finetuning/                    # Fine-tuning datasets
│   │   ├── company_a/                 # Synthetic company A
│   │   ├── company_b/                 # Synthetic company B
│   │   └── template.csv               # Data format template
│   └── literature/                    # Literature data
│       ├── cost_overrun_stats.csv     # From Flyvbjerg et al.
│       ├── delay_stats.csv            # From Merrow
│       └── risk_stats.csv             # From Loch & Kavadias
│
├── models/                            # Trained models
│   ├── pretrained/
│   │   ├── ppo_agent_final.pt         # Final pretrained agent
│   │   ├── ppo_agent_best.pt          # Best validation agent
│   │   └── training_log.json          # Training metrics
│   ├── finetuned/
│   │   ├── company_a_agent.pt
│   │   └── company_b_agent.pt
│   └── baselines/
│       └── mip_solutions.pkl          # Cached MIP solutions
│
├── results/                           # Experiment results
│   ├── baseline_comparison/
│   │   ├── results.csv                # Raw results
│   │   ├── summary.json               # Summary statistics
│   │   └── plots/                     # Generated plots
│   ├── scalability/
│   │   ├── timing_results.csv
│   │   └── plots/
│   ├── sensitivity/
│   │   ├── uncertainty_sweep.csv
│   │   ├── budget_sweep.csv
│   │   └── plots/
│   ├── finetuning/
│   │   ├── company_a_results.csv
│   │   ├── company_b_results.csv
│   │   └── plots/
│   └── interpretability/
│       ├── shap_values.csv
│       ├── attention_weights.csv
│       └── plots/
│
├── docs/                              # Documentation
│   ├── README.md                      # Documentation index
│   ├── installation.md                # Installation guide
│   ├── quickstart.md                  # Quick start tutorial
│   ├── api-reference.md               # API documentation
│   ├── finetuning-guide.md            # Fine-tuning tutorial
│   ├── deployment.md                  # Deployment guide
│   ├── architecture.md                # System architecture
│   ├── experiments.md                 # Reproducing experiments
│   └── contributing.md                # Contribution guidelines
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_baselines.py
│   ├── test_backend.py
│   └── test_evaluation.py
│
├── scripts/                           # Utility scripts
│   ├── setup_environment.sh           # Environment setup
│   ├── download_dependencies.sh       # Download Gurobi, etc.
│   ├── generate_instances.py          # Generate synthetic data
│   ├── train_agent.py                 # Training script
│   ├── run_experiments.sh             # Run all experiments
│   ├── generate_paper_figures.sh      # Generate all figures
│   └── deploy_system.sh               # Deploy to cloud
│
├── docker/                            # Docker configurations
│   ├── Dockerfile.backend             # Backend container
│   ├── Dockerfile.frontend            # Frontend container
│   ├── Dockerfile.experiments         # Experiments container
│   └── nginx.conf                     # Nginx configuration
│
└── notebooks/                         # Jupyter notebooks
    ├── 01_data_exploration.ipynb      # Explore synthetic data
    ├── 02_baseline_analysis.ipynb     # Analyze baselines
    ├── 03_rl_training_analysis.ipynb  # Training curves
    ├── 04_results_visualization.ipynb # Visualize results
    └── 05_interpretability.ipynb      # Interpretability analysis
```
---

## Key Design Decisions

### 1. Shared Figures Between Paper and Thesis
Use symbolic links to avoid duplication:
```bash
cd thesis/figures
ln -s ../../paper/figures/* .
```
This ensures figures are generated once and used in both documents.

### 2. Bilingual Bibliography Management
- `paper/references.bib`: English sources for journal article
- `thesis/references-fa.bib`: Persian sources (translated titles, Persian authors)
- `thesis/references-en.bib`: English sources with Persian annotations

### 3. Reproducibility First
- All experiments are scripts in `experiments/`
- All results saved to `results/` with timestamps
- All figures generated programmatically from results
- No manual data manipulation

### 4. Modular System Architecture
- `src/` contains all core logic
- `backend/` is a thin API wrapper around `src/`
- `frontend/` is completely decoupled
- Each component can be tested independently

### 5. Docker Deployment
```yaml
# docker-compose.yml
services:
  backend:
    build: ./docker/Dockerfile.backend
    ports: ["8000:8000"]
    volumes: ["./models:/app/models"]
  
  frontend:
    build: ./docker/Dockerfile.frontend
    ports: ["3000:3000"]
  
  nginx:
    image: nginx:alpine
    ports: ["80:80"]
    volumes: ["./docker/nginx.conf:/etc/nginx/nginx.conf"]
```
---

## LaTeX Structure Details

### English Paper (EJOR Format)

```latex
% paper/main.tex
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm,algorithmic}
\usepackage{hyperref}
\usepackage{natbib}

\title{Reinforcement Learning for Real-Time Portfolio Budgeting Under Uncertainty: 
       A Deployable Framework}
\author{Your Name \\ Department of Industrial Engineering \\ University Name}

\begin{document}
\maketitle

\begin{abstract}
% 150-200 words
\end{abstract}

\section{Introduction}
\input{sections/01-introduction}

\section{Literature Review}
\input{sections/02-literature}

\section{Problem Formulation}
\input{sections/03-problem-formulation}

\section{Methodology}
\input{sections/04-methodology}

\section{Computational Experiments}
\input{sections/05-experiments}

\section{System Architecture and Deployment}
\input{sections/06-system-architecture}

\section{Managerial Insights}
\input{sections/07-managerial-insights}

\section{Conclusion}
\input{sections/08-conclusion}

\bibliographystyle{elsarticle-harv}
\bibliography{references}

\appendix
\input{sections/appendix}

\end{document}
```
### Persian Thesis (XePersian)

```latex
% thesis/main.tex
\documentclass[12pt,a4paper]{report}
\usepackage{xepersian}
\settextfont{XB Niloofar}  % یا فونت دانشگاه شما
\setdigitfont{XB Niloofar}

\input{config/preamble}
\input{config/commands}

\begin{document}

\input{frontmatter/title-page}
\input{frontmatter/abstract-fa}
\input{frontmatter/abstract-en}
\input{frontmatter/acknowledgments}
\input{frontmatter/dedication}
\tableofcontents
\listoffigures
\listoftables

\input{chapters/01-introduction}
\input{chapters/02-literature}
\input{chapters/03-methodology}
\input{chapters/04-implementation}
\input{chapters/05-experiments}
\input{chapters/06-system}
\input{chapters/07-conclusion}

\appendix
\input{appendices/appendix-a-code}
\input{appendices/appendix-b-data}
\input{appendices/appendix-c-results}
\input{appendices/appendix-d-manual}

\bibliographystyle{ieeetr-fa}  % یا استایل دانشگاه
\bibliography{references-fa,references-en}

\end{document}
```
---

## File Creation Priority

### Phase 1: Foundation (Weeks 1-3)

✓ Create directory structure
✓ Setup Git repository
✓ Write README.md
✓ Create requirements.txt
✓ Implement src/environment/
✓ Implement src/baselines/greedy.py
✓ Write experiments/01_environment_validation.py
✓ Generate data/synthetic/instances/

### Phase 2: Baselines (Weeks 4-7)

✓ Implement src/baselines/mip_solver.py
✓ Implement src/baselines/stochastic_prog.py
✓ Implement src/baselines/perfect_info.py
✓ Write experiments/02_baseline_comparison.py
✓ Generate results/baseline_comparison/

### Phase 3: RL Agent (Weeks 8-11)

✓ Implement src/agents/ppo_agent.py
✓ Implement src/agents/network.py
✓ Implement src/agents/trainer.py
✓ Write experiments/03_rl_training.py
✓ Train models/pretrained/ppo_agent_final.pt

### Phase 4: System Development (Weeks 12-16)

✓ Implement src/backend/
✓ Implement frontend/
✓ Write docker/Dockerfile.*
✓ Write docs/installation.md
✓ Write docs/quickstart.md

### Phase 5: Analysis (Weeks 17-19)

✓ Write experiments/04-07_*.py
✓ Generate all results/
✓ Implement src/evaluation/interpretability.py
✓ Write experiments/08_generate_figures.py
✓ Generate paper/figures/

### Phase 6: Writing (Weeks 20-22)

✓ Write paper/sections/*.tex
✓ Write thesis/chapters/*.tex
✓ Generate paper/submission/manuscript.pdf
✓ Generate thesis/submission/thesis-final.pdf

---

## Makefile for Common Tasks

```makefile
# Makefile

.PHONY: help install train test deploy paper thesis clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make data          - Generate synthetic data"
	@echo "  make train         - Train RL agent"
	@echo "  make experiments   - Run all experiments"
	@echo "  make figures       - Generate paper figures"
	@echo "  make test          - Run unit tests"
	@echo "  make backend       - Start backend server"
	@echo "  make frontend      - Start frontend dev server"
	@echo "  make deploy        - Deploy full system"
	@echo "  make paper         - Compile English paper"
	@echo "  make thesis        - Compile Persian thesis"
	@echo "  make clean         - Clean generated files"

install:
	pip install -r requirements.txt
	cd frontend && npm install

data:
	python scripts/generate_instances.py

train:
	python scripts/train_agent.py

experiments:
	bash scripts/run_experiments.sh

figures:
	bash scripts/generate_paper_figures.sh

test:
	pytest tests/

backend:
	cd src/backend && uvicorn app:app --reload

frontend:
	cd frontend && npm run dev

deploy:
	docker-compose up -d

paper:
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

thesis:
	cd thesis && xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	cd paper && rm -f *.aux *.log *.bbl *.blg *.out
	cd thesis && rm -f *.aux *.log *.bbl *.blg *.out
```
---

## Git Strategy

### .gitignore
```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/

# Data (too large for Git)
data/synthetic/instances/
models/pretrained/*.pt
models/finetuned/*.pt
results/

# LaTeX
*.aux
*.log
*.bbl
*.blg
*.out
*.toc
*.lof
*.lot
*.synctex.gz

# Frontend
frontend/node_modules/
frontend/dist/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

### Git LFS for Large Files
bash
git lfs track "*.pt"
git lfs track "*.pkl"
git lfs track "data/literature/*.csv"
```
---

## Documentation Files to Create

### docs/installation.md
```markdown
# Installation Guide

## Prerequisites
- Python 3.9+
- Node.js 16+
- Gurobi license (academic free)
- Docker (optional)

## Step 1: Clone Repository
...

## Step 2: Install Python Dependencies
...

## Step 3: Install Frontend Dependencies
...

## Step 4: Setup Gurobi
...

## Step 5: Verify Installation
...
```
### docs/finetuning-guide.md
```markdown
# Fine-Tuning Guide

## Data Format
Your company data should be a CSV with columns:
- project_id
- cost_estimate
- actual_cost
- return_estimate
- actual_return
- duration_estimate
- actual_duration
- risk_score

## Step 1: Prepare Data
...

## Step 2: Upload to System
...

## Step 3: Trigger Fine-Tuning
...

## Step 4: Evaluate Results
...
```
---

## Timeline Integration

| Week | Tasks | Files Created |
|------|-------|---------------|
| 1-3 | Environment + validation | `src/environment/`, `data/synthetic/` |
| 4-7 | Baselines | `src/baselines/`, `results/baseline_comparison/` |
| 8-11 | RL training | `src/agents/`, `models/pretrained/` |
| 12-16 | System development | `src/backend/`, `frontend/`, `docker/` |
| 17-19 | Experiments + analysis | `experiments/`, `results/`, `paper/figures/` |
| 20-22 | Writing | `paper/sections/`, `thesis/chapters/` |

---

## Critical Success Factors

1. **Start with structure**: Create all directories and placeholder files in Week 1
2. **Incremental commits**: Commit after each experiment, not at the end
3. **Shared figures**: Generate once, use in both paper and thesis
4. **Reproducibility**: Every result must be script-generated
5. **Documentation**: Write docs as you build, not at the end

This structure supports a complete, reproducible, publishable project with deployable software.
