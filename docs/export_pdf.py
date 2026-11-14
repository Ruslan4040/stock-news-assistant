"""Export docs/REPORT.md to docs/REPORT.pdf (Markdown -> HTML -> headless Chrome).

Run: python docs/export_pdf.py
Needs Google Chrome or Chromium, and internet access once to load mermaid.js for the diagram.
"""

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

DOCS = Path(__file__).resolve().parent
CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]
RENDER_WAIT_MS = 15000  # time for mermaid.js to draw before printing
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"
STYLE = """
body { font: 10.5pt/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
       max-width: 46em; margin: 0 auto; }
h1 { font-size: 20pt; } h2 { font-size: 14pt; margin-top: 1.6em; } h3 { font-size: 11.5pt; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 8.8pt; }
th, td { border: 1px solid #c3c2b7; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: #f0efec; } code { font-size: 9pt; background: #f0efec; padding: 0 3px; }
pre { background: #f9f9f7; padding: 8px; overflow-wrap: anywhere; white-space: pre-wrap; }
pre.mermaid { background: none; text-align: center; } blockquote { color: #52514e; }
@page { size: A4; margin: 16mm 14mm; } h2, h3 { break-after: avoid; } tr { break-inside: avoid; }
"""


def find_chrome() -> str:
    """Return the path of an installed Chrome or Chromium."""
    for candidate in CHROME_PATHS:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    sys.exit("Chrome or Chromium was not found; install one or print REPORT.md another way.")


def to_html(markdown: str) -> str:
    """Convert Markdown to a full HTML page; mermaid blocks become live diagrams."""
    body = MarkdownIt("commonmark").enable("table").render(markdown)
    body = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        lambda m: f'<pre class="mermaid">{html.unescape(m.group(1))}</pre>',
        body, flags=re.S,
    )
    script = (f'<script src="{MERMAID}"></script>'
              "<script>mermaid.initialize({startOnLoad:true});</script>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLE}</style></head>"
            f"<body>{body}{script}</body></html>")


def main() -> None:
    """Render the report and print it to PDF."""
    source, target = DOCS / "REPORT.md", DOCS / "REPORT.pdf"
    with tempfile.TemporaryDirectory() as folder:
        page = Path(folder) / "report.html"
        page.write_text(to_html(source.read_text(encoding="utf-8")), encoding="utf-8")
        command = [find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                   f"--virtual-time-budget={RENDER_WAIT_MS}", f"--print-to-pdf={target}",
                   page.as_uri()]
        subprocess.run(command, check=True, capture_output=True)
    print(f"wrote {target.relative_to(DOCS.parent)} ({target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
