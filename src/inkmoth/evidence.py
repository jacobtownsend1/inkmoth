"""Deterministic provenance checks; these do not prove that a passage entails a claim."""
import re
from collections import Counter

from .schemas import Assessment, Fact, Findings, RejectedFact, SubQuestionResult
from .sources import SourceRegistry


def validate_findings(findings: Findings, registry: SourceRegistry, actor: str) -> tuple[Findings, list[str]]:
    facts, warnings = [], []
    rejected = list(findings.rejected)
    for fact in findings.facts:
        evidence = [e for e in fact.evidence
                    if e.source_id in fact.source_ids and registry.supports(actor, e.source_id, e.quote)]
        ids = sorted({e.source_id for e in evidence})
        if not ids or set(ids) != set(fact.source_ids) or len(evidence) != len(fact.evidence):
            missing = sorted(set(fact.source_ids) - set(ids))
            invalid = [i for i, e in enumerate(fact.evidence, 1)
                       if e.source_id not in fact.source_ids or not registry.supports(actor, e.source_id, e.quote)]
            rejected.append(RejectedFact(fact=fact, reason=
                f"No complete provenance: unsupported source IDs {missing}; invalid/unseen quote indices {invalid}"))
            continue
        facts.append(fact.model_copy(update={"source_ids": ids, "evidence": evidence}))
    if rejected:
        warnings.append(f"Evidence validation rejected {len(rejected)} claim(s); see findings.rejected diagnostics")
    # Rejected claim text is diagnostic data only. Never copy it into gaps/prompts.
    return Findings(summary=" ".join(f.claim for f in facts), facts=facts,
                    gaps=list(dict.fromkeys(g for g in findings.gaps if not g.startswith("Unverified claim omitted:"))),
                    rejected=rejected), warnings


SINGLE_SITE = "†"
SINGLE_SITE_LEGEND = "† Single-site evidence: every source cited there is on one website."
_SOURCES_HEADING = r"(?im)^\s*(?:#{1,6}\s+)?(?:sources|references|bibliography)\s*:?\s*$"
_FENCED = re.compile(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$|^~~~[^\n]*\n.*?^~~~[^\n]*$")


def _outside_code(pattern: str, text: str):
    """First match of `pattern` that is not inside a fenced code block, or None."""
    fences = [m.span() for m in _FENCED.finditer(text)]
    return next((m for m in re.finditer(pattern, text) if not any(a <= m.start() < b for a, b in fences)), None)


def facts_by_ref(results: list[SubQuestionResult]) -> dict[str, Fact]:
    """Internal fact refs (r<sub-question>f<n>) used by the assessor; never shown to the writer."""
    return {f"r{r.sub_question.id}f{i}": f for r in results if r.findings for i, f in enumerate(r.findings.facts, 1)}


def disagreement_sides(assessment: Assessment | None, results: list[SubQuestionResult]) -> list[dict]:
    """Validated contradictions as summary + each side's claim and source IDs (the writer's citation space)."""
    facts = facts_by_ref(results)
    return [{"summary": c.summary, "sides": [{"claim": facts[r].claim, "source_ids": facts[r].source_ids}
                                              for r in c.fact_refs if r in facts]}
            for c in (assessment.contradictions if assessment else [])]


def ensure_disagreements(text: str, assessment: Assessment | None, results: list[SubQuestionResult]) -> str:
    """Contradictions must reach the reader: add a cited section when the writer or reviewer left it out."""
    items = disagreement_sides(assessment, results)
    if not items or _outside_code(r"(?im)^#{1,6}\s+disagreements\b", text):
        return text
    lines = ["## Disagreements", ""] + [
        f"- {d['summary'].rstrip(' .;:')}: " + "; ".join(s["claim"] + " " + "".join(f"[{i}]" for i in s["source_ids"])
                                          for s in d["sides"]) for d in items]
    m = _outside_code(_SOURCES_HEADING, text)
    at = m.start() if m else len(text)
    return text[:at].rstrip() + "\n\n" + "\n".join(lines) + "\n\n" + text[at:]


def rejection_notes(results: list[SubQuestionResult], registry: SourceRegistry) -> str:
    """What failed quote validation so far, as source IDs and counts only. Rejected claim text never goes
    into prompts (it may carry injected instructions); the reason is fixed text, not the model's."""
    quoted, unread = Counter(), 0
    for r in results:
        for rejected in (r.findings.rejected if r.findings else []):
            fetched = [i for i in dict.fromkeys(rejected.fact.source_ids) if (s := registry.get(i)) and s.fetched]
            quoted.update(fetched)
            unread += not fetched
    notes = []
    if quoted:
        ids = ", ".join(f"[{i}] ({n})" for i, n in sorted(quoted.items()))
        notes.append(f"Earlier claims citing {ids} were rejected because their quotes did not match the fetched "
                     "text verbatim. If you use these sources, re-read them and copy short quotes exactly.")
    if unread:
        notes.append(f"{unread} earlier claim(s) cited sources that were never fetched; fetch a page before relying on it.")
    return " ".join(notes)


def frame_report(md: str, assumptions: list[str], clarifications: str) -> str:
    """Put what the user clarified, or what was assumed for them, directly under the report title."""
    line = (f"**Clarified by the user:** {' '.join(clarifications.split())}" if clarifications
            else "**Assumptions:** " + "; ".join(assumptions) if assumptions else "")
    if not line:
        return md
    title, _, rest = md.partition("\n")
    return f"{title}\n\n{line}\n{rest}" if title.startswith("# ") else f"{line}\n\n{md}"


def cited_ids(results: list[SubQuestionResult]) -> set[int]:
    return {sid for r in results if r.findings for f in r.findings.facts for sid in f.source_ids}


def finish_markdown(text: str, registry: SourceRegistry, allowed: set[int]) -> tuple[str, list[str]]:
    text = text.replace("\x00", "")
    # Preserve code verbatim: Python indexing such as a[0] is not a citation.
    code = []
    def protect(match):
        code.append(match.group(0))
        return f"\x00CODE{len(code) - 1}\x00"
    text = re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$|^~~~[^\n]*\n.*?^~~~[^\n]*$|`[^`\n]+`", protect, text)
    # The model doesn't control the bibliography, including source URLs or titles.
    text = re.split(_SOURCES_HEADING, text)[0].strip()
    # Only the application marks single-site evidence; drop any marker or legend the model wrote.
    text = re.sub(r"[ \t]*\[single (?:source|site)\]|(?<=\])†", "", text, flags=re.I)
    text = re.sub(r"(?m)^\s*†[^\n]*\n?", "", text).rstrip()
    warnings = []
    used = set()

    def citation(match):
        if any(len(number) > 9 for number in re.findall(r"\d+", match.group(1))):
            warnings.append("Unsupported oversized citation ID")
            return "[unverified]"
        ids = []
        for part in re.split(r"\s*[,;]\s*", match.group(1)):
            bounds = re.split(r"\s*[-–]\s*", part)
            if len(bounds) == 2:
                first, last = map(int, bounds)
                if not 0 <= last - first < 32:
                    warnings.append(f"Unsupported citation range: {match.group(0)}")
                    return "[unverified]"
                ids.extend(range(first, last + 1))
            else:
                ids.append(int(part))
        valid = [sid for sid in ids if sid in allowed and registry.get(sid)]
        used.update(valid)
        if len(valid) != len(ids):
            warnings.append(f"Unsupported citation removed: {match.group(0)}")
        return "".join(f"[{sid}]" for sid in valid) or "[unverified]"

    text = re.sub(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\](?!\()", citation, text)
    # Numeric Markdown links supplied by the model are replaced by the registry IDs.
    text = re.sub(r"\[(\d+)\]\([^\n)]*\)", citation, text)
    # Corroboration is computed, not requested: a citation group whose sources share one website gets a †.
    text = re.sub(r"\[\d+\](?:[ \t]*\[\d+\])*", lambda m: m.group(0) + (
        SINGLE_SITE if len(registry.hosts(int(i) for i in re.findall(r"\d+", m.group(0)))) == 1 else ""), text)
    if SINGLE_SITE in text:
        text += "\n\n" + SINGLE_SITE_LEGEND
    if not used:
        warnings.append("No verified citations in the generated answer")
    bibliography = registry.table(used) or "No verified sources cited."
    text = re.sub(r"\x00CODE(\d+)\x00", lambda m: code[int(m.group(1))], text)
    return text + "\n\n## Sources\n" + bibliography, list(dict.fromkeys(warnings))


def fallback_report(question: str, results: list[SubQuestionResult]) -> str:
    parts = [f"# Research: {question}", "Partial research; a complete synthesis is unavailable."]
    for result in results:
        parts.append(f"## {result.sub_question.question}")
        if result.findings:
            parts.extend(f"- {fact.claim} " + " ".join(f"[{i}]" for i in fact.source_ids)
                         for fact in result.findings.facts)
            if result.findings.gaps or result.findings.rejected:
                parts.append("Some requested details remain unestablished; see the structured research diagnostics.")
        elif result.error:
            parts.append(f"Incomplete: {result.error}")
    return "\n\n".join(parts)
