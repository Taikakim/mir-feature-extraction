# plots/latent_analysis/findings.py
"""FINDINGS.md section-overwrite and PROGRESS.md helpers."""
import re
from datetime import datetime
from pathlib import Path

from plots.latent_analysis.config import DATA_DIR

FINDINGS_PATH = DATA_DIR.parent / "FINDINGS.md"
PROGRESS_PATH = DATA_DIR.parent / "PROGRESS.md"

_FINDINGS_TEMPLATE = """\
# Latent Feature Analysis Findings

## Run info
- Date: {date}
- N crops: {n_crops}
- N features: {n_features} (after encoding + variance drop)
- Features dropped (low variance): {dropped}

<!-- script-01-start -->
## Script 01 — Aggregate Correlations
*(not yet run)*
<!-- script-01-end -->

<!-- script-02-start -->
## Script 02 — PCA
*(not yet run)*
<!-- script-02-end -->

<!-- script-03-start -->
## Script 03 — Latent Cross-Correlation
*(not yet run)*
<!-- script-03-end -->

<!-- script-04-start -->
## Script 04 — Temporal
*(not yet run)*
<!-- script-04-end -->

## Open questions / next steps
*(add manual notes here — this section is never auto-overwritten)*
"""


def overwrite_findings_section(content: str, script_id: str, new_body: str) -> str:
    """
    Replace the content between <!-- script-{id}-start --> and <!-- script-{id}-end -->
    markers with new_body. Raises ValueError if markers are absent.
    """
    start_marker = f"<!-- script-{script_id}-start -->"
    end_marker   = f"<!-- script-{script_id}-end -->"
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )
    if not re.search(pattern, content):
        raise ValueError(
            f"FINDINGS.md section markers not found for script-{script_id}. "
            f"Expected '{start_marker}' ... '{end_marker}'."
        )
    replacement = f"{start_marker}\n{new_body.rstrip()}\n{end_marker}"
    return re.sub(pattern, replacement, content)


def init_findings(n_crops: int, n_features: int, dropped: list):
    """Create FINDINGS.md from template if it doesn't exist."""
    if FINDINGS_PATH.exists():
        return
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = _FINDINGS_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        n_crops=n_crops,
        n_features=n_features,
        dropped=", ".join(dropped) if dropped else "none",
    )
    FINDINGS_PATH.write_text(text)


def update_findings_section(script_id: str, body: str):
    """Overwrite a named section in FINDINGS.md. Idempotent."""
    if not FINDINGS_PATH.exists():
        raise RuntimeError("FINDINGS.md not initialised — call init_findings() first.")
    content = FINDINGS_PATH.read_text()
    content = overwrite_findings_section(content, script_id, body)
    FINDINGS_PATH.write_text(content)


def update_progress(script_id: str, message: str):
    """Append a timestamped line to PROGRESS.md."""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] Script {script_id}: {message}\n"
    with open(PROGRESS_PATH, "a") as f:
        f.write(line)
    print(line.strip())
