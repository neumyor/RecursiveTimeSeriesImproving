from __future__ import annotations

from pathlib import Path

from harnessing_ts.state.read_docx import extract_docx


LAYOUT_DIRS = (
    "user",
    "user/loop-memory",
    "data/raw",
    "data/processed",
    "references",
    "knowledge_base",
    "knowledge_base/tables",
    "knowledge_base/indexes",
    "artifacts",
    "plots",
    "tools",
    "tools/generated",
    "runs/iterations",
    "reports",
    "reports/iterations",
    "logs/nodes",
    "state/nodes",
    "training",
)


def ensure_workspace_layout(root: Path) -> None:
    for rel in LAYOUT_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    ensure_reference_text_derivatives(root)


def ensure_reference_text_derivatives(root: Path) -> None:
    references = root / "references"
    if not references.exists():
        return
    for path in references.iterdir():
        if path.is_file() and path.suffix.lower() == ".docx":
            write_reference_text_derivative(root, path)


def write_reference_text_derivative(root: Path, path: Path) -> str | None:
    if path.suffix.lower() != ".docx":
        return None
    target = path.with_suffix(path.suffix + ".txt")
    if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return str(target.relative_to(root))
    try:
        text = extract_docx(path)
    except Exception:
        return None
    target.write_text(text + "\n", encoding="utf-8")
    return str(target.relative_to(root))
