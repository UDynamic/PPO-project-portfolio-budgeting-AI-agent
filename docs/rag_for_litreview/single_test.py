import fitz

path = r"F:\Career\00. Univesity\Masters\05. Master's thesis\PPO-project-portfolio-budgeting-AI-agent\docs\rag_for_litreview\lit_archive_failed\00. A comprehensive mathematical model for resource-constrained multiobjective project-Noted- 2024.09.12 , .pdf"

doc = fitz.open(path)
for i, page in enumerate(doc):
    text = page.get_text().strip()
    if len(text) > 100:
        print(f"\n--- Page {i+1} sample ---")
        print(text[:500])
        break