"""Check the eval dataset against the PDFs it makes claims about.

Ground truth is the one part of an eval suite nothing else can validate: a wrong
expected page or a typo'd figure quietly turns a passing system into a failing
score. This asserts, for every case, that at least one `patterns` entry really
does occur on at least one of the `pages` it names, reading the PDFs directly.

Patterns are written to match a model's prose (e.g. "2.9 sigma") as well as the
paper's typography (e.g. "2.9σ"), so only one alternative needs to match.

    python -m evals.verify_dataset
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_assistant.loader import load_pdf  # noqa: E402

DATASET = Path(__file__).with_name("dataset.yaml")


def _page_text(data_dir: Path) -> dict[tuple[str, int], str]:
    """Map (filename, page) -> normalised page text for the whole corpus."""
    text: dict[tuple[str, int], str] = {}
    for pdf in sorted(data_dir.glob("*.pdf")):
        for doc in load_pdf(pdf):
            text[(pdf.name, doc.metadata["page"])] = re.sub(r"\s+", " ", doc.page_content)
    return text


def main() -> int:
    spec = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    data_dir = PROJECT_ROOT / "data"
    pages = _page_text(data_dir)
    available = {name for name, _ in pages}

    failures: list[str] = []
    checked = 0

    # Papers on disk that no case exercises. Not a failure, but worth surfacing:
    # adding papers changes what retrieval competes against, so existing
    # expectations may no longer be unambiguous.
    exercised = {
        source
        for case in spec["cases"]
        for source in (case.get("sources") or ([case["source"]] if case.get("source") else []))
    }
    uncovered = sorted(available - exercised)

    for case in spec["cases"]:
        case_id = case["id"]
        sources = case.get("sources") or ([case["source"]] if case.get("source") else [])
        patterns = case.get("patterns") or []

        for source in sources:
            if source not in available:
                failures.append(f"{case_id}: source {source} not found in {data_dir}")

        if not patterns:
            continue  # out-of-corpus cases assert nothing about content

        if case.get("verify_against") == "filenames":
            # Inventory answers come from the list_papers tool, so the ground truth
            # is the set of filenames, not anything inside the PDFs.
            checked += 1
            if not any(
                re.search(pattern, name, re.IGNORECASE)
                for pattern in patterns
                for name in available
            ):
                failures.append(f"{case_id}: none of {patterns} match any filename")
            continue

        if not sources:
            # open_fact: no paper is named, so verify the information exists
            # *somewhere* in the corpus rather than in a specific file.
            checked += 1
            hits = {
                pattern: sorted(
                    {name for (name, _), text in pages.items()
                     if re.search(pattern, text, re.IGNORECASE)}
                )
                for pattern in patterns
            }
            unmatched = [pattern for pattern, names in hits.items() if not names]
            if len(unmatched) == len(patterns):
                failures.append(
                    f"{case_id}: none of {patterns} found anywhere in the corpus"
                )
            continue

        declared = case.get("pages")
        for source in sources:
            candidate_pages = declared or [p for (name, p) in pages if name == source]
            haystacks = [
                pages[(source, page)] for page in candidate_pages if (source, page) in pages
            ]
            missing_pages = [p for p in (declared or []) if (source, p) not in pages]
            if missing_pages:
                failures.append(f"{case_id}: {source} has no page(s) {missing_pages}")
            if not haystacks:
                continue

            checked += 1
            matched = [
                pattern
                for pattern in patterns
                if any(re.search(pattern, h, re.IGNORECASE) for h in haystacks)
            ]
            if not matched:
                failures.append(
                    f"{case_id}: none of {patterns} found in {source} "
                    f"pages {declared or 'any'}"
                )

    print(f"Checked {checked} source/pattern claim(s) across {len(spec['cases'])} case(s).")
    print(f"Corpus on disk: {len(available)} paper(s); {len(exercised)} exercised by the dataset.")
    if uncovered:
        print(f"\n{len(uncovered)} paper(s) on disk with no eval case:")
        for name in uncovered:
            print(f"  - {name}")
        print(
            "  These still compete for retrieval, so check that existing questions "
            "remain unambiguous."
        )
    if failures:
        print(f"\n{len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All ground-truth claims verified against the PDFs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
