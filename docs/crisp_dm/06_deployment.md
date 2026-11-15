# CRISP-DM 6. Deployment
Purpose: Streamlit app with a trace panel, README run instructions, `docs/REPORT.md`, demo script.
Gate: the app runs from a clean clone by following the README; the UI checklist passes.

## What is deployed
| Item | Where |
|---|---|
| Streamlit app: sidebar request, Advice / Simulator / Evals tabs, trace panel | `app/streamlit_app.py`, charts in `app/charts.py` |
| Setup and run instructions, diagram, business rules, results table | `README.md` |
| Short report (and PDF export) | `docs/REPORT.md`, `docs/REPORT.pdf`, `docs/export_pdf.py` |
| Demo script for the video (at most 10 minutes) | `docs/demo_script.md` |

`docs/REPORT.md` and `docs/demo_script.md` are rendered from `docs/templates/` by `python -m evals.render_docs`, which fills every number from `evals/results/`.

## UI checklist (run by `python -m evals.eval_ui` on the demo cases in `config.yaml`)
1. The app runs without an exception.
2. The three tabs render.
3. The advice card (action, horizon, confidence, news used) equals the pipeline output.
4. The four price features equal the numbers the advisor received.
5. The headline table equals the collected headlines with their relevance and sentiment.
6. The trace panel lists exactly the LLM calls of this request.
7. The Simulator and Evals tabs show the stored results.

## Log

## 2026-11-15 — Clean-clone check, report, demo script
- What was done:
  - Fresh `git clone` into an empty folder; followed the README: `uv venv`, `uv pip install -r requirements.txt`, `.env` and the Kaggle CSVs copied in by hand (the two manual steps), `src.data.download`, `src.data.prepare`, `ruff`, `pytest`, `streamlit run`, `evals.eval_ui`.
  - `docs/REPORT.md`, `docs/REPORT.pdf` and `docs/demo_script.md` rendered from `docs/templates/` with every number taken from `evals/results/`.
- Decisions & why:
  - The UI checklist in the clone ran with an empty LLM cache, so it used the real API. Rejected: copying `.cache/llm` into the clone, which would hide a broken API setup.
  - LLM cache and pipeline runs stay out of git (`data/`, `.cache/`); the app's Simulator and Evals tabs read the committed results files, so they work in a fresh clone.
- Artifacts: `docs/REPORT.md`, `docs/REPORT.pdf`, `docs/demo_script.md`, `README.md` (PDF link).
- Evidence:
  - Clean clone: install ok; download 300/300 Finnhub windows; `ruff` clean; `pytest` 138 passed; `/_stcore/health` → ok; UI checklist 21 of 21 checks, 3 of 3 cases.
  - Sets rebuilt in the clone have the same 300 decision ids and the same headlines as the original; price features differ by at most about 1e-6 (Yahoo adjusted-price float noise), below the 2-decimal rounding of the advisor input.
- Gate status: yes.
- Open questions for the human: label review (see `05_evaluation.md`); score set D after the close of 2026-11-17.
