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
