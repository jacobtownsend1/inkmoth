"""Summarize saved runs. Structural checks do not establish factual correctness."""

import argparse
import json
import re
from pathlib import Path
from inkmoth.sources import normalize_quote


def audit(path):
    report = json.loads(path.read_text())
    sources = {s["id"]: s for s in report["sources"]}
    facts = [f for r in report["results"] if r["findings"] for f in r["findings"]["facts"]]
    citations = {int(n) for n in re.findall(r"\[(\d+)\]", report["report_md"].split("## Sources")[0])}
    unmatched = []
    for fact in facts:
        for evidence in fact["evidence"]:
            source = sources.get(evidence["source_id"], {})
            quote = normalize_quote(evidence["quote"])
            if not any(quote in normalize_quote(excerpt) for excerpt in source.get("excerpts", [])):
                unmatched.append({"claim": fact["claim"], "source_id": evidence["source_id"]})
    return {
        "case": path.parent.name,
        "status": report["status"],
        **report["stats"],
        "words": len(report["report_md"].split()),
        "facts": len(facts),
        "warnings": len(report["warnings"]),
        "fetched_sources": sum(s["fetched"] for s in sources.values()),
        "cited_sources": len(citations),
        "unknown_citation_ids": sorted(citations - sources.keys()),
        "unmatched_quotes": unmatched,
        "rejected_claims": sum(len(r["findings"].get("rejected", [])) for r in report["results"] if r["findings"]),
        "review_corrections": (report.get("review") or {}).get("issues", []),
        "coverage": (report.get("assessment") or {}).get("coverage", []),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    results = [audit(p) for p in sorted(Path(args.directory).glob("*/report.json"))]
    print(json.dumps(results, indent=2))
