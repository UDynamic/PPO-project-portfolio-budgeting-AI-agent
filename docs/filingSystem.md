# Complete Project Structure: RL Portfolio Budgeting System + Publications

## Overview
This structure supports:
1. English Q1 journal article (LaTeX)
2. Persian M.Sc. thesis documentation (XePersian/LaTeX)
3. Full system implementation (RL agent, baselines, backend, dual frontend)
4. Reproducible experiments and data
5. Documentation and deployment guides
6. Strategic research planning and archival

---

## Root Directory Structure

```
rl-portfolio-budgeting/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── strategy/                    
├── literatureArchive/                 
├── mscn/                              
│
├── paper/
│   ├── main.tex
│   ├── references.bib
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
│   ├── figures/
│   │   ├── architecture-diagram.pdf
│   │   ├── speedup-comparison.pdf
│   │   ├── quality-comparison.pdf
│   │   ├── scalability-plot.pdf
│   │   ├── sensitivity-analysis.pdf
│   │   ├── attention-heatmap.pdf
│   │   └── ui-screenshots/
│   │       ├── web-interface.png
│   │       └── desktop-interface.png
│   ├── tables/
│   │   ├── baseline-comparison.tex
│   │   ├── scalability-results.tex
│   │   └── finetuning-results.tex
│   ├── supplementary/
│   │   ├── supplementary.tex
│   │   ├── hyperparameters.tex
│   │   ├── additional-experiments.tex
│   │   └── code-snippets.tex
│   └── submission/
│       ├── manuscript.pdf
│       ├── cover-letter.tex
│       ├── response-to-reviewers.tex
│       └── highlights.txt
│
├── thesis/
│   ├── main.tex
│   ├── references-fa.bib
│   ├── references-en.bib
│   ├── config/
│   │   ├── preamble.tex
│   │   ├── commands.tex
│   │   └── university-style.sty
│   ├── frontmatter/
│   │   ├── title-page.tex
│   │   ├── abstract-fa.tex
│   │   ├── abstract-en.tex
│   │   ├── acknowledgments.tex
│   │   ├── dedication.tex
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
│   ├── figures/                       # symlink -> ../paper/figures
│   └── submission/
│       ├── thesis-final.pdf
│       └── defense-presentation.pdf
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── Makefile
    ├── docker-compose.yml
    ├── environment.yml
    ├── environment/
    │   ├── __init__.py
    │   ├── generator.py
    │   ├── distributions.py
    │   ├── calibration.py
    │   ├── validator.py
    │   └── portfolio.py
    ├── agents/
    │   ├── __init__.py
    │   ├── ppo_agent.py
    │   ├── network.py
    │   ├── attention.py
    │   ├── trainer.py
    │   └── finetuner.py
    ├── baselines/
    │   ├── __init__.py
    │   ├── greedy.py
    │   ├── mip_solver.py
    │   ├── stochastic_prog.py
    │   └── perfect_info.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py
    │   ├── comparator.py
    │   ├── sensitivity.py
    │   └── interpretability.py
    ├── backend/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── routes/
    │   │   ├── __init__.py
    │   │   ├── allocate.py
    │   │   ├── finetune.py
    │   │   ├── evaluate.py
    │   │   └── health.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── request.py
    │   │   └── response.py
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── inference.py
    │   │   ├── finetuning.py
    │   │   └── storage.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── validation.py
    │       └── logging.py
    ├── frontend/
    │   ├── web/
    │   │   ├── app/
    │   │   │   ├── layout.tsx
    │   │   │   ├── page.tsx
    │   │   │   ├── upload/
    │   │   │   │   └── page.tsx
    │   │   │   ├── allocate/
    │   │   │   │   └── page.tsx
    │   │   │   ├── results/
    │   │   │   │   └── page.tsx
    │   │   │   ├── finetune/
    │   │   │   │   └── page.tsx
    │   │   │   └── compare/
    │   │   │       └── page.tsx
    │   │   ├── components/
    │   │   │   ├── ui/
    │   │   │   ├── ProjectTable.tsx
    │   │   │   ├── AllocationChart.tsx
    │   │   │   ├── MetricsCard.tsx
    │   │   │   └── FileUpload.tsx
    │   │   ├── lib/
    │   │   │   ├── api.ts
    │   │   │   ├── utils.ts
    │   │   │   └── types.ts
    │   │   ├── public/
    │   │   ├── styles/
    │   │   │   └── globals.css
    │   │   ├── package.json
    │   │   ├── next.config.js
    │   │   ├── tsconfig.json
    │   │   └── tailwind.config.js
    │   └── desktop/
    │       ├── app/
    │       │   ├── __init__.py
    │       │   ├── main_window.py
    │       │   ├── widgets/
    │       │   │   ├── __init__.py
    │       │   │   ├── project_table.py
    │       │   │   ├── allocation_chart.py
    │       │   │   ├── metrics_card.py
    │       │   │   └── file_upload.py
    │       │   ├── dialogs/
    │       │   │   ├── __init__.py
    │       │   │   ├── upload_dialog.py
    │       │   │   ├── finetune_dialog.py
    │       │   │   └── settings_dialog.py
    │       │   ├── models/
    │       │   │   ├── __init__.py
    │       │   │   └── data_models.py
    │       │   ├── services/
    │       │   │   ├── __init__.py
    │       │   │   └── api_client.py
    │       │   └── assets/
    │       │       ├── icons/
    │       │       ├── images/
    │       │       └── styles/
    │       │           └── main.qss
    │       ├── resources/
    │       │   └── resources.qrc
    │       ├── main.py
    │       ├── requirements.txt
    │       └── build.spec
    ├── experiments/
    │   ├── 01_environment_validation.py
    │   ├── 02_baseline_comparison.py
    │   ├── 03_rl_training.py
    │   ├── 04_scalability_analysis.py
    │   ├── 05_sensitivity_analysis.py
    │   ├── 06_finetuning_study.py
    │   ├── 07_interpretability.py
    │   └── 08_generate_figures.py
    ├── data/
    │   ├── synthetic/
    │   │   ├── parameters.json
    │   │   ├── instances/
    │   │   │   ├── train/
    │   │   │   ├── val/
    │   │   │   └── test/
    │   │   └── validation_stats.json
    │   ├── finetuning/
    │   │   ├── company_a/
    │   │   ├── company_b/
    │   │   └── template.csv
    │   └── literature/
    │       ├── cost_overrun_stats.csv
    │       ├── delay_stats.csv
    │       └── risk_stats.csv
    ├── models/
    │   ├── pretrained/
    │   │   ├── ppo_agent_final.pt
    │   │   ├── ppo_agent_best.pt
    │   │   └── training_log.json
    │   ├── finetuned/
    │   │   ├── company_a_agent.pt
    │   │   └── company_b_agent.pt
    │   └── baselines/
    │       └── mip_solutions.pkl
    ├── results/
    │   ├── baseline_comparison/
    │   │   ├── results.csv
    │   │   ├── summary.json
    │   │   └── plots/
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
    ├── docs/
    │   ├── README.md
    │   ├── installation.md
    │   ├── quickstart.md
    │   ├── api-reference.md
    │   ├── finetuning-guide.md
    │   ├── deployment.md
    │   ├── architecture.md
    │   ├── experiments.md
    │   ├── frontend-development.md
    │   └── contributing.md
    ├── tests/
    │   ├── __init__.py
    │   ├── test_environment.py
    │   ├── test_agents.py
    │   ├── test_baselines.py
    │   ├── test_backend.py
    │   ├── test_evaluation.py
    │   └── test_frontend_api.py
    ├── scripts/
    │   ├── setup_environment.sh
    │   ├── download_dependencies.sh
    │   ├── generate_instances.py
    │   ├── train_agent.py
    │   ├── run_experiments.sh
    │   ├── generate_paper_figures.sh
    │   ├── build_desktop_app.sh
    │   └── deploy_system.sh
    ├── docker/
    │   ├── Dockerfile.backend
    │   ├── Dockerfile.frontend-web
    │   ├── Dockerfile.experiments
    │   └── nginx.conf
    └── utils/
        ├── __init__.py
        ├── data_loader.py
        ├── visualization.py
        ├── logger.py
        └── reproducibility.py
```
---

## Key Design Decisions

### 1. Research Planning & Archival (`strategy/`, `literatureArchive/`, `mscn/`)
Centralized home for all strategic planning, decision rationale, and conversation history. The three sibling folders are yours to populate freely — kept separate from the codebase intentionally.

### 2. Dual Frontend Architecture

**Next.js (`src/frontend/web/`)** — build this first:
- Online deployment, reviewer demos, PhD committee access
- Next.js 14 App Router + TypeScript + Tailwind + shadcn/ui
- Deploy to Vercel or Docker + Nginx

**PyQt (`src/frontend/desktop/`)** — refactor from web after:
- Offline/enterprise use, distributable `.exe`/`.app`
- PyQt6 + PyInstaller for packaging
- Shares the same backend API — zero duplication of business logic

### 3. Shared Figures (Paper ↔ Thesis)
bash
cd thesis/figures
ln -s ../../paper/figures/* .
Generate once in `paper/figures/`, use in both documents.

### 4. Bilingual Bibliography
- `paper/references.bib` — English sources for journal
- `thesis/references-en.bib` — English sources with Persian annotations
- `thesis/references-fa.bib` — Persian sources

### 5. Reproducibility
Every result is script-generated from `src/experiments/` and saved to `src/results/`. No manual data manipulation anywhere.

---

## LaTeX Templates

### English Paper (`paper/main.tex`)
```latex
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx,booktabs}
\usepackage{algorithm,algorithmic}
\usepackage{hyperref,natbib}

\title{Reinforcement Learning for Real-Time Portfolio Budgeting Under Uncertainty:
       A Deployable Decision Support Framework}
\author{Your Name \\ Department of Industrial Engineering \\ University Name}

\begin{document}
\maketitle
\begin{abstract}% 150-200 words\end{abstract}

\section{Introduction}             \input{sections/01-introduction}
\section{Literature Review}        \input{sections/02-literature}
\section{Problem Formulation}      \input{sections/03-problem-formulation}
\section{Methodology}              \input{sections/04-methodology}
\section{Computational Experiments}\input{sections/05-experiments}
\section{System Architecture}      \input{sections/06-system-architecture}
\section{Managerial Insights}      \input{sections/07-managerial-insights}
\section{Conclusion}               \input{sections/08-conclusion}

\bibliographystyle{elsarticle-harv}
\bibliography{references}
\appendix
\input{sections/appendix}
\end{document}
```
### Persian Thesis (`thesis/main.tex`)
```latex
\documentclass[12pt,a4paper]{report}
\usepackage{xepersian}
\settextfont{XB Niloofar}
\setdigitfont{XB Niloofar}

\input{config/preamble}
\input{config/commands}

\begin{document}
\input{frontmatter/title-page}
\input{frontmatter/abstract-fa}
\input{frontmatter/abstract-en}
\input{frontmatter/acknowledgments}
\input{frontmatter/dedication}
\tableofcontents\listoffigures\listoftables

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

\bibliographystyle{ieeetr-fa}
\bibliography{references-fa,references-en}
\end{document}
```
---

## `src/Makefile`
```makefile
.PHONY: help install data train experiments figures test \
        backend frontend-web frontend-desktop build-desktop \
        deploy paper thesis clean

help:
	@echo "install            Install all dependencies"
	@echo "data               Generate synthetic data"
	@echo "train              Train RL agent"
	@echo "experiments        Run all experiments"
	@echo "figures            Generate paper figures"
	@echo "test               Run unit tests"
	@echo "backend            Start FastAPI backend"
	@echo "frontend-web       Start Next.js dev server"
	@echo "frontend-desktop   Run PyQt desktop app"
	@echo "build-desktop      Build PyQt executable"
	@echo "deploy             Deploy via Docker"
	@echo "paper              Compile English paper (pdflatex)"
	@echo "thesis             Compile Persian thesis (xelatex)"
	@echo "clean              Remove generated files"

install:
	pip install -r ../requirements.txt
	cd frontend/web && npm install
	cd frontend/desktop && pip install -r requirements.txt

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
	cd backend && uvicorn app:app --reload

frontend-web:
	cd frontend/web && npm run dev

frontend-desktop:
	cd frontend/desktop && python main.py

build-desktop:
	bash scripts/build_desktop_app.sh

deploy:
	docker-compose up -d

paper:
	cd ../../paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

thesis:
	cd ../../thesis && xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	cd ../../paper && rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
	cd ../../thesis && rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
	cd frontend/web && rm -rf .next
```
---

## `src/docker-compose.yml`
```yaml
services:
  backend:
    build:
      context: ..
      dockerfile: src/docker/Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models

  frontend-web:
    build:
      context: ./frontend/web
      dockerfile: ../../docker/Dockerfile.frontend-web
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
      - frontend-web
```
---

## `src/environment.yml`
```yaml
name: rl-portfolio
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scipy
  - matplotlib
  - seaborn
  - jupyter
  - pip:
    - torch
    - stable-baselines3
    - gymnasium
    - fastapi
    - uvicorn
    - pydantic
    - pulp
    - pyomo
    - shap
    - pytest
    - black
    - ruff
```
---

## .gitignore

```
# Python
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/

# Large files (use Git LFS instead)
src/data/synthetic/instances/
src/models/pretrained/*.pt
src/models/finetuned/*.pt
src/results/

# LaTeX
*.aux *.log *.bbl *.blg *.out
*.toc *.lof *.lot *.synctex.gz

# Next.js
src/frontend/web/node_modules/
src/frontend/web/.next/
src/frontend/web/out/
src/frontend/web/.env*.local

# PyQt build
src/frontend/desktop/build/
src/frontend/desktop/dist/

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db

### Git LFS
bash
git lfs track "*.pt"
git lfs track "*.pkl"
git lfs track "src/data/literature/*.csv"
```
---

## Implementation Timeline

| Week  | Phase               | Key Deliverables                                                      |
|-------|---------------------|-----------------------------------------------------------------------|
| 1–3   | Foundation          | `src/environment/`, `src/data/synthetic/`, `strategy/` setup         |
| 4–7   | Baselines           | `src/baselines/`, `src/results/baseline_comparison/`                  |
| 8–11  | RL Agent            | `src/agents/`, `src/models/pretrained/`                               |
| 12–16 | System Development  | `src/backend/`, `src/frontend/web/`, `src/frontend/desktop/`         |
| 17–19 | Analysis            | `src/experiments/04-08`, `src/results/`, `paper/figures/`            |
| 20–22 | Writing             | `paper/sections/`, `thesis/chapters/`, submission files              |

---

## Critical Success Factors

1. Create all directories and placeholder files in Week 1
2. Commit after each experiment — not at the end
3. Generate figures once in `paper/figures/`, symlink from `thesis/`
4. Every result must be reproducible from a script
5. Write `src/docs/` as you build, not at the end
6. Build Next.js first, refactor to PyQt — share the backend API throughout
