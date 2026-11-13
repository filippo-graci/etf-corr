"""Convert the reference PDFs in ``references/pdf/`` to Markdown in ``references/md/``.
Run with ``uv run python scripts/papers_to_md.py`` (seconds; needs ``pdftotext`` from poppler).

One file per paper, one ``## Page N`` section per PDF page, text as extracted (reading order, no
layout), with a header naming the version and source from ``references/sources.tsv``. Page numbers
are those of the PDF file, which for working papers differ from the published article's.
Equations and tables come out as plain text and may be garbled: check the PDF before quoting one.

The PDFs and their text are research copies for this project only and are git-ignored; only
``sources.tsv`` and ``README.md`` in ``references/`` are committed.
"""

import csv
import hashlib
import subprocess

from etfcorr.config import find_repo_root


def main() -> int:
    root = find_repo_root() / "references"
    (root / "md").mkdir(exist_ok=True)
    with open(root / "sources.tsv", encoding="utf-8") as f:
        sources = {row["key"]: row for row in csv.DictReader(f, delimiter="\t")}
    for pdf in sorted((root / "pdf").glob("*.pdf")):
        text = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if not text.split():  # a page scan without a text layer: nothing to convert
            print(f"{pdf.stem}: no text layer, skipped (read the PDF)")
            continue
        pages = text.split("\f")
        if pages and not pages[-1].strip():
            pages = pages[:-1]
        meta = sources.get(pdf.stem, {"version": "unknown", "url": "unknown"})
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        body = [
            f"# {pdf.stem}",
            "",
            f"- Version: {meta['version']}",
            f"- Source: {meta['url']}",
            f"- PDF sha256: {digest}",
            f"- Pages: {len(pages)} (page numbers below are those of the PDF file)",
            "- Research copy for this project only; not for redistribution.",
            "",
        ]
        for number, page in enumerate(pages, start=1):
            body += [f"## Page {number}", "", page.strip(), ""]
        (root / "md" / f"{pdf.stem}.md").write_text("\n".join(body), encoding="utf-8")
        print(f"{pdf.stem}: {len(pages)} pages, {len(text.split())} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
