import fitz  # PyMuPDF
import os
import shutil
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer

# --- Config ---
PDF_FOLDER = "./lit_archive"
PASS_FOLDER = "./lit_archive_passed"
FAIL_FOLDER = "./lit_archive_failed"

MIN_CHARS = 500
MIN_PAGES_READABLE = 0.5

LOG_FILE_NAME = "history_log.md"
HASH_FILE_NAME = "processed_hashes.txt"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Ensure folders exist ---
os.makedirs(PASS_FOLDER, exist_ok=True)
os.makedirs(FAIL_FOLDER, exist_ok=True)


# --- Hashing ---
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_hashes(folder):
    path = os.path.join(folder, HASH_FILE_NAME)
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f.readlines())


def append_hash(folder, file_hash):
    path = os.path.join(folder, HASH_FILE_NAME)
    with open(path, "a") as f:
        f.write(file_hash + "\n")


def is_duplicate(file_hash):
    return file_hash in load_hashes(PASS_FOLDER) or file_hash in load_hashes(FAIL_FOLDER)


# --- Logging ---
def append_log(folder, log_entry):
    log_path = os.path.join(folder, LOG_FILE_NAME)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n\n")


def write_run_header():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"# RUN — {timestamp}\n"

    append_log(PASS_FOLDER, header)
    append_log(FAIL_FOLDER, header)


def format_log(filename, overall_pass, t1, t2, t3):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "PASS ✅" if overall_pass else "FAIL ❌"

    return f"""
## {timestamp} — {filename}

**Result:** {status}

### Test 1 - Extraction
- Passed: {t1.get('passed')}
- Readable Ratio: {t1.get('readable_ratio')}
- Total Chars: {t1.get('total_chars')}
- Error: {t1.get('error')}

### Test 2 - Text Quality
- Passed: {t2.get('passed')}
- Alpha Ratio: {t2.get('alpha_ratio')}
- Avg Word Length: {t2.get('avg_word_length')}
- Reason: {t2.get('reason')}

### Test 3 - Embedding
- Passed: {t3.get('passed')}
- Embedding Dim: {t3.get('embedding_dim')}
- Reason: {t3.get('reason')}
"""


# --- Tests ---

def test_extraction(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            readable_pages = 0
            total_chars = 0

            for page in doc:
                text = page.get_text().strip()
                if len(text) > MIN_CHARS:
                    readable_pages += 1
                    total_chars += len(text)

        ratio = readable_pages / total_pages if total_pages > 0 else 0

        return {
            "passed": ratio >= MIN_PAGES_READABLE,
            "total_pages": total_pages,
            "readable_pages": readable_pages,
            "readable_ratio": round(ratio, 2),
            "total_chars": total_chars,
            "error": None,
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_text_quality(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            sample_text = ""
            count = 0

            for page in doc:
                text = page.get_text().strip()
                if len(text) > MIN_CHARS:
                    sample_text += text
                    count += 1
                if count >= 3:
                    break

        if not sample_text:
            return {"passed": False, "reason": "No readable text found"}

        words = sample_text.split()
        total_words = len(words)

        alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
        alpha_ratio = len(alpha_words) / total_words if total_words > 0 else 0

        avg_word_len = (
            sum(len(w) for w in words) / total_words if total_words > 0 else 0
        )

        passed = alpha_ratio > 0.45 and 3 < avg_word_len < 12

        return {
            "passed": passed,
            "alpha_ratio": round(alpha_ratio, 2),
            "avg_word_length": round(avg_word_len, 2),
            "sample_words": total_words,
            "reason": None if passed else "Low alpha ratio or abnormal word length",
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


def test_embedding(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            sample_text = ""

            for page in doc:
                text = page.get_text().strip()
                if len(text) > MIN_CHARS:
                    sample_text = text[:1000]
                    break

        if not sample_text:
            return {"passed": False, "reason": "No text to embed"}

        embedding = embedder.encode([sample_text])

        return {
            "passed": True,
            "embedding_dim": len(embedding[0]),
            "reason": None,
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


# --- Main Runner ---

def run_diagnostics(pdf_folder):
    files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    print(f"\nFound {len(files)} PDFs to test\n")
    print("=" * 70)

    # --- Write run headers ---
    write_run_header()

    passed_count = 0
    failed_count = 0
    skipped_duplicates = 0

    for filename in files:
        path = os.path.join(pdf_folder, filename)

        # --- Hash check ---
        file_hash = compute_file_hash(path)

        if is_duplicate(file_hash):
            print(f"\nSkipping duplicate: {filename}")
            skipped_duplicates += 1
            os.remove(path)  # remove duplicate from archive
            continue

        print(f"\nTesting: {filename}")

        t1 = test_extraction(path)
        t2 = test_text_quality(path) if t1["passed"] else {"passed": False, "reason": "Skipped"}
        t3 = test_embedding(path) if t2["passed"] else {"passed": False, "reason": "Skipped"}

        overall_pass = t1["passed"] and t2["passed"] and t3["passed"]

        print(f"  → Overall: {'PASS' if overall_pass else 'FAIL'}")

        dest_folder = PASS_FOLDER if overall_pass else FAIL_FOLDER
        dest_path = os.path.join(dest_folder, filename)

        # --- Move file ---
        shutil.move(path, dest_path)

        # --- Save hash ---
        append_hash(dest_folder, file_hash)

        # --- Log ---
        log_entry = format_log(filename, overall_pass, t1, t2, t3)
        append_log(dest_folder, log_entry)

        if overall_pass:
            passed_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_count} passed | {failed_count} failed | {skipped_duplicates} duplicates skipped")


# --- Run ---
if __name__ == "__main__":
    run_diagnostics(PDF_FOLDER)