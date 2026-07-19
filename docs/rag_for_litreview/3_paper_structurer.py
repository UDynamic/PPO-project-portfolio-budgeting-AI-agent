"""
paper_structurer.py
-------------------
Extracts structured knowledge from academic PDFs using Groq (FREE LLaMA3 model).
Outputs thesis-specific JSON covering literature review, environment calibration,
and RL agent design needs.

Handles:
  - highly_relevant/ and valuable/ folders
  - Skips already-structured PDFs (checks by filename)
  - Safe to re-run anytime — only processes new additions

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
# LOAD .env
# ─────────────────────────────────────────────
def load_env():
    current = Path(__file__).resolve()
    # walks up looking for .env — works regardless of nesting depth
    for parent in current.parents:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            print(f"✓ Loaded .env from: {candidate}")
            return
    print("⚠ No .env file found — GROQ_API_KEY must be set as system env variable")

load_env()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_KEY  = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "llama-3.3-70b-versatile"

# Both folders to process
TARGET_FOLDERS = [
    "./lit_archive_passed/highly_relevant",
    "./lit_archive_passed/valuable",
]

MAX_CHARS     = 6000
SLEEP_BETWEEN = 0.3


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

Return ONLY valid JSON. No explanations. No markdown.

each metadata has a meaning:
- portfolio level: portfolio or project level,
- decision type: scheduling, selection, resource allocation or budgeting
- stochastic cash flows: uncertain or random cash inflow and outflow
- contract mechanics: advanced payment, retention, milestone payment, earned value management
- termination settlement: completion delay penalty or completion cost overrun penalty
- method: LP, stochastic programming or RL
- model type: LP, stochastic programming or RL

"""

def build_prompt(text: str) -> str:
    return f"""Extract structured information.

Return ONLY this JSON schema:

{{
  "metadata": {{
    "title_guess": "",
    "year_guess": "",
    "problem": "",
    "portfolio level": "",
    "decision type": "",
    "sequential decisions": ""
    "stochastic cash flows": "",
    "contract mechanics": "",
    "termination settlement": "",
    "method": "",
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
    "dynamic horizon": null
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
# STEP 3 — API Call (Groq)
# ─────────────────────────────────────────────
def call_api(prompt: str, retry: int = 0) -> str | None:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens":  1200
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 429:
            wait = 30 if retry == 0 else 60
            print(f"  ⚠ Rate limited — waiting {wait}s...")
            time.sleep(wait)
            return call_api(prompt, retry=retry + 1)

        if response.status_code != 200:
            print(f"  ✗ API error {response.status_code}: {response.text[:200]}")
            return None

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return None


# ─────────────────────────────────────────────
# STEP 4 — Robust JSON extraction
# ─────────────────────────────────────────────
def extract_json(raw: str) -> dict | None:
    if not raw:
        return None

    # Attempt 1 — direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Attempt 2 — strip markdown fences
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Attempt 3 — find outermost JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Attempt 4 — fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────
# STEP 5 — Failure logging
# ─────────────────────────────────────────────
def save_failure(failed_folder: str, filename: str, reason: str, raw: str = ""):
    os.makedirs(failed_folder, exist_ok=True)
    fail_path = os.path.join(failed_folder, filename.replace(".pdf", "_FAILED.txt"))
    with open(fail_path, "w", encoding="utf-8") as f:
        f.write(f"File:      {filename}\n")
        f.write(f"Reason:    {reason}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write("Raw LLM response:\n")
        f.write(raw)
    print(f"  → Failure logged: {fail_path}")


# ─────────────────────────────────────────────
# PROCESS ONE FOLDER
# ─────────────────────────────────────────────
def process_folder(base_folder: str) -> dict:
    """
    Processes all PDFs in base_folder.
    Structured JSONs go to base_folder/structured/
    Failed logs  go to base_folder/structured_failed/

    Skip logic:
      - If JSON already exists for this filename → skip (already done)
      - If FAILED log exists → skip (don't retry automatically)
        Delete the _FAILED.txt manually to force a retry.
    """
    output_folder = os.path.join(base_folder, "structured")
    failed_folder = os.path.join(base_folder, "structured_failed")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(failed_folder, exist_ok=True)

    # Collect PDFs — exclude subfolders like structured/ itself
    files = sorted([
        f for f in os.listdir(base_folder)
        if f.endswith(".pdf")
        and os.path.isfile(os.path.join(base_folder, f))
    ])

    if not files:
        print(f"  No PDFs found in {base_folder}")
        return {"passed": 0, "failed": 0, "skipped": 0, "total": 0}

    passed = failed = skipped = 0

    for i, filename in enumerate(files, 1):
        pdf_path   = os.path.join(base_folder, filename)
        json_path  = os.path.join(output_folder, filename.replace(".pdf", ".json"))
        fail_path  = os.path.join(failed_folder, filename.replace(".pdf", "_FAILED.txt"))

        print(f"  [{i}/{len(files)}] {filename[:65]}")

        # ── SKIP: already structured ──
        if os.path.exists(json_path):
            print("    → Already structured — skipping\n")
            skipped += 1
            continue

        # ── SKIP: previously failed (delete _FAILED.txt to retry) ──
        if os.path.exists(fail_path):
            print("    → Previously failed — skipping (delete _FAILED.txt to retry)\n")
            skipped += 1
            continue

        # ── EXTRACT TEXT ──
        try:
            text = extract_key_sections(pdf_path)
        except Exception as e:
            print(f"    ✗ Extraction error: {e}")
            save_failure(failed_folder, filename, f"Extraction error: {e}")
            failed += 1
            continue

        if len(text) < 200:
            print("    ✗ Not enough text extracted")
            save_failure(failed_folder, filename, "Insufficient text")
            failed += 1
            continue

        print(f"    ✓ Extracted {len(text)} chars")

        # ── CALL LLM ──
        prompt = build_prompt(text)
        raw    = call_api(prompt)

        if not raw:
            save_failure(failed_folder, filename, "No API response")
            failed += 1
            continue

        # ── PARSE JSON ──
        parsed = extract_json(raw)

        if parsed is None:
            print("    ✗ JSON parsing failed")
            save_failure(failed_folder, filename, "JSON parse failed", raw)
            failed += 1
            continue

        # ── SAVE ──
        parsed["_source"] = {
            "filename": filename,
            "folder":   base_folder,
            "time":     datetime.now().isoformat(),
            "model":    MODEL
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"    ✓ Saved → {json_path}\n")
        passed += 1

        time.sleep(SLEEP_BETWEEN)

    return {"passed": passed, "failed": failed, "skipped": skipped, "total": len(files)}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_structuring():
    print(f"\n{'='*60}")
    print(f"Paper Structurer")
    print(f"Model  : {MODEL}")
    print(f"Folders: {len(TARGET_FOLDERS)}")
    print(f"{'='*60}\n")

    totals = {"passed": 0, "failed": 0, "skipped": 0, "total": 0}

    for folder in TARGET_FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠ Folder not found, skipping: {folder}\n")
            continue

        label = os.path.basename(folder).upper()
        print(f"── {label} ──────────────────────────────────────")

        stats = process_folder(folder)

        for k in totals:
            totals[k] += stats[k]

        print(f"  Subtotal → passed: {stats['passed']} | "
              f"failed: {stats['failed']} | skipped: {stats['skipped']}\n")

    print(f"{'='*60}")
    print(f"TOTAL SUMMARY")
    print(f"  Structured : {totals['passed']}")
    print(f"  Failed     : {totals['failed']}")
    print(f"  Skipped    : {totals['skipped']}")
    print(f"  Total PDFs : {totals['total']}")
    print(f"{'='*60}")

    if totals["failed"] > 0:
        print("\nTo retry failed files: delete the corresponding _FAILED.txt and re-run.")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: GROQ_API_KEY not found in .env")
        exit(1)

    run_structuring()