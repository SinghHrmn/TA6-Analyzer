# TA6 Analyser

AI-assisted document analysis and enquiry generation for UK residential
conveyancing seller-disclosure (TA6) forms.
MSc Applied AI dissertation — H. Singh (4436296), LSBU.

The system reads a TA6 (digital PDF **or** scanned paper), extracts its content,
detects problems (missing/incomplete answers **and** cross-document
contradictions against the title report, EPC, searches and planning), and drafts
solicitor-style follow-up enquiries. It is **assistive**: a solicitor reviews
every output.

---

## Project layout

```
ta6_analyser/
├── ta6/                       core package (importable)
│   ├── pipeline.py            extract → detect (rules + structured) → generate
│   ├── nli.py                 cross-document contradiction detection (LLM/NLI)
│   ├── schema_builder.py      canonical schema from the editable template
│   └── generator.py           labelled synthetic-data generator
├── scripts/                   runnable experiments
│   ├── run_real.py            run the pipeline on one real form
│   ├── run_full_detect.py     rules + NLI on one form (demo)
│   ├── evaluate.py            detection P/R/F1 on the synthetic set
│   └── eval_ocr_vs_digital.py digital-vs-OCR comparison (results table)
├── webapp/
│   └── app.py                 Flask upload/analyse/download UI + run logging
├── data/
│   └── ta6_schema.json        generated canonical schema (454 fields)
├── pyproject.toml             package + dependencies
├── requirements.txt
└── README.md
```

---

## 1. Prerequisites (system tools — not pip)

The pipeline shells out to two standard tools for scanned PDFs:

| Tool | Provides | Install |
|------|----------|---------|
| **Poppler** | `pdftotext`, `pdftoppm` | macOS: `brew install poppler` · Ubuntu: `sudo apt install poppler-utils` |
| **Tesseract** | OCR | macOS: `brew install tesseract` · Ubuntu: `sudo apt install tesseract-ocr` |

For the contradiction-detection stage you also need **one** LLM backend
(see §4). The default backend needs nothing installed.

---

## 2. Set up an isolated environment

```bash
cd ta6_analyser

python3 -m venv .venv           # create the virtual environment
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e .                # installs the package + all Python deps
```

`pip install -e .` puts the `ta6` package on your path, so every script and the
web app can `import ta6` from anywhere while the environment is active.

Deactivate any time with `deactivate`.

---

## 3. Run it

All commands assume the venv is active (`source .venv/bin/activate`).

```bash
# Analyse one real TA6 (auto-detects digital vs scanned)
python scripts/run_real.py "/path/to/TA6.pdf"

# Rules + cross-document contradiction demo on one form
python scripts/run_full_detect.py

# Rebuild the canonical schema from the editable 6th-edition template
python -m ta6.schema_builder "/path/to/EDITABLE TA6 6th edition.pdf"

# Generate a labelled synthetic dataset (digital + scan, with faults)
python -m ta6.generator --n 100 --seed 7 --pdf --out synth_out

# Detection precision/recall/F1 on the synthetic set
python scripts/evaluate.py synth_out

# Digital-vs-OCR comparison (the results-chapter table)
python scripts/eval_ocr_vs_digital.py 100
```

### Web app

```bash
export TA6_NLI_BACKEND=ollama          # optional; see §4
python webapp/app.py                    # → http://127.0.0.1:5000
```

Upload a TA6 (and optionally the title report / EPC / search / planning PDFs) to
see extracted content, flagged issues with severity, and draft enquiries. Every
run is appended to `webapp/runs.jsonl` (an evaluation-harness log).

---

## 4. Contradiction-detection backend (NLI)

Selected with the `TA6_NLI_BACKEND` environment variable. The code supports three
interchangeable backends:

| `TA6_NLI_BACKEND` | Needs | Notes |
|-------------------|-------|-------|
| *(unset)* / `stub` | nothing | Offline lexical **baseline**. Runs everywhere; report it as a baseline, not the contribution. |
| `ollama` | [Ollama](https://ollama.com) + a local model | **Recommended.** Free, private — real client text never leaves your machine (cleanest for GDPR). |
| `anthropic` | `pip install -e ".[anthropic]"` + `ANTHROPIC_API_KEY` | Highest quality; small cloud cost. Do **not** send unredacted client data. |

Ollama setup:

```bash
ollama pull llama3.1            # or qwen2.5:7b for lower RAM
export TA6_NLI_BACKEND=ollama
export OLLAMA_MODEL=llama3.1    # optional; this is the default
```

---

## 5. Data & ethics

- **Synthetic data is primary.** Fabricated forms are not personal data, so they
  carry no GDPR exposure. Real forms are used only for validation.
- Keep any real forms out of version control (see `.gitignore` → `data/real/`),
  redact them, and never send unredacted forms to a cloud API. The `ollama`
  backend keeps everything local.

---

## 6. Reproducibility

All stochastic steps take a fixed `--seed`. `pip install -e .` pins the package;
`requirements.txt` lists the Python dependencies; system tools and versions are
listed in §1. Re-running any script with the same seed reproduces the reported
numbers (subject to OCR/model nondeterminism, which is noted where it applies).
