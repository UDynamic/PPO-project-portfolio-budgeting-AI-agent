"""
paper_structurer.py
-------------------
Extracts structured knowledge from academic PDFs using Groq (FREE LLaMA3 model).
Outputs thesis-specific JSON covering literature review, environment calibration,
and RL agent design needs.

Usage:
    python paper_structurer.py

Requirements:
    pip install pymupdf requests python-dotenv
"""

import fitz  # PyMuPDF
import os
import json
import re
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


# ─────────────────────────────────────────────
# LOAD .env FROM /docs/ (one level above this script)
# ─────────────────────────────────────────────
def load_env():
    current = Path(__file__).resolve()

    # expected structure:
    # root/
    #   docs/
    #     .env
    #     rag_for_litreview/
    #       paper_structurer.py

    env_path = current.parent.parent.parent / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f".env not found at: {env_path}")

    load_dotenv(env_path)


load_env()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_KEY = os.getenv("GROQ_API_KEY")

BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ BEST FREE MODEL ON GROQ
MODEL = "llama-3.3-70b-versatile"

BASE_FOLDER    = "./lit_archive_passed/highly_relevant"
OUTPUT_FOLDER  = os.path.join(BASE_FOLDER, "structured")
FAILED_FOLDER  = os.path.join(BASE_FOLDER, "structured_failed")

MAX_CHARS      = 6000
SLEEP_BETWEEN  = 0.3  # Groq is fast → can reduce delay slightly

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — Extract key sections from PDF
# ─────────────────────────────────────────────
def extract_key_sections(pdf_path: str) -> str:
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]

    full_text  = "\n".join(pages)
    text_lower = full_text.lower()

    def get_section(keyword: str, length: int = 2000) -> str:
        idx = text_lower.find(keyword)
        if idx == -1:
            return ""
        return full_text[idx: idx + length]

    abstract   = get_section("abstract",     2000)
    intro      = get_section("introduction", 2000)
    conclusion = get_section("conclusion",   2000)

    combined = "\n\n".join(filter(None, [abstract, intro, conclusion])).strip()

    if len(combined) < 300:
        combined = "\n".join(pages[:2])

    return combined[:MAX_CHARS]


# ─────────────────────────────────────────────
# STEP 2 — Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a research assistant helping with a Master's thesis on:
Reinforcement Learning for Project Portfolio Management Budgeting.

Return ONLY valid JSON. No explanations. No markdown."""

def build_prompt(text: str) -> str:
    return f"""Extract structured information.

Return ONLY this JSON schema:

{{
  "metadata": {{
    "title_guess": "",
    "year_guess": "",
    "problem": "",
    "method": "",
    "domain": "",
    "level": "",
    "model_type": "",
    "data_type": ""
  }},
  "calibration_params": {{
    "cpi_stats": null,
    "spi_stats": null,
    "cost_overrun": null,
    "schedule_overrun": null,
    "project_type": null,
    "sample_size": null,
    "scurve_params": null,
    "efficiency_params": null,
    "horizon": null
  }},
  "thesis_relevance": {{
    "supports_environment_design": false,
    "supports_rl_agent": false,
    "supports_problem_motivation": false,
    "supports_calibration": false,
    "key_finding": "",
    "gap_identified": null
  }},
  "key_variables": []
}}

TEXT:
{text}"""


# ─────────────────────────────────────────────
# STEP 3 — API CALL (Groq)
# ─────────────────────────────────────────────
def call_api(prompt: str) -> str | None:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1200
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 429:
            print("  ⚠ Rate limited — waiting 5 seconds...")
            time.sleep(5)
            return call_api(prompt)

        if response.status_code != 200:
            print(f"  ✗ API error {response.status_code}: {response.text[:200]}")
            return None

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return None


# ─────────────────────────────────────────────
# STEP 4 — JSON extraction
# ─────────────────────────────────────────────
def extract_json(raw: str) -> dict | None:
    if not raw:
        return None

    try:
        return json.loads(raw)
    except:
        pass

    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(fixed)
    except:
        pass

    return None


# ─────────────────────────────────────────────
# STEP 5 — Failure logging
# ─────────────────────────────────────────────
def save_failure(filename: str, reason: str, raw_response: str = ""):
    fail_path = os.path.join(FAILED_FOLDER, filename.replace(".pdf", "_FAILED.txt"))

    with open(fail_path, "w", encoding="utf-8") as f:
        f.write(f"File: {filename}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(raw_response)

    print(f"  → Failure logged: {fail_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_structuring():
    files = sorted([
        f for f in os.listdir(BASE_FOLDER)
        if f.endswith(".pdf")
    ])

    if not files:
        print("No PDFs found in:", BASE_FOLDER)
        return

    print(f"\n{'='*60}")
    print(f"Found {len(files)} PDFs")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")

    passed = failed = skipped = 0

    for i, filename in enumerate(files, 1):
        pdf_path  = os.path.join(BASE_FOLDER, filename)
        json_path = os.path.join(OUTPUT_FOLDER, filename.replace(".pdf", ".json"))

        print(f"[{i}/{len(files)}] {filename[:70]}")

        if os.path.exists(json_path):
            print("  → Skipped\n")
            skipped += 1
            continue

        try:
            text = extract_key_sections(pdf_path)
        except Exception as e:
            save_failure(filename, str(e))
            failed += 1
            continue

        if len(text) < 200:
            save_failure(filename, "Too little text")
            failed += 1
            continue

        prompt = build_prompt(text)
        raw = call_api(prompt)

        if not raw:
            save_failure(filename, "No response")
            failed += 1
            continue

        parsed = extract_json(raw)

        if parsed is None:
            save_failure(filename, "JSON failed", raw)
            failed += 1
            continue

        parsed["_source"] = {
            "file": filename,
            "time": datetime.now().isoformat(),
            "model": MODEL
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print("  ✓ Done\n")
        passed += 1

        time.sleep(SLEEP_BETWEEN)

    print("\nDONE")
    print(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: GROQ_API_KEY missing in .env")
        exit(1)

    run_structuring()