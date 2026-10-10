"""Pydantic models shared by tools, agents, pipeline and CLI. Planner/researcher outputs are
structured-output targets, so keep field descriptions: the model sees them."""
from pydantic import BaseModel, Field, model_validator
from typing import Literal


class SubQuestion(BaseModel):
    id: int = Field(description="1-based index, unique across the whole research run")
    question: str = Field(min_length=1, max_length=2000, description="A focused, searchable sub-question")
    why: str = Field(description="One sentence on how answering it serves the main question")
    search_hints: list[str] = Field(default_factory=list, description="2-4 concrete search queries to start with")


class Plan(BaseModel):
    sub_questions: list[SubQuestion] = Field(max_length=100)


class Coverage(BaseModel):
    question: str = Field(description="One explicit part of the ORIGINAL user question")
    status: Literal["answered", "partial", "unanswered"]
    fact_refs: list[str] = Field(default_factory=list, description="Supporting fact refs such as r1f2; never gap text")


class ClarificationCheck(BaseModel):
    needs_clarification: bool = Field(description="True only if different reasonable readings would change the plan materially")
    questions: list[str] = Field(default_factory=list, description="At most 3 short questions for the user")
    assumptions: list[str] = Field(default_factory=list,
                                   description="What you will assume for each open point if the user does not answer")

    @model_validator(mode="after")
    def _cap(self):  # an over-long answer is trimmed, not rejected: losing the whole check is worse
        self.questions = self.questions[:3]
        # Assumptions for a clear question carry no information and, in practice, leak remembered "facts"
        # into every later prompt; keep them only when there was a real ambiguity to resolve.
        self.assumptions = self.assumptions[:5] if self.needs_clarification else []
        return self


class Contradiction(BaseModel):
    summary: str = Field(max_length=500, description="What the sources disagree about, stated neutrally")
    fact_refs: list[str] = Field(default_factory=list, max_length=8,
                                 description="Fact refs such as r1f2 for each conflicting side; at least two")


class Assessment(BaseModel):
    sufficient: bool = False
    gaps: list[str] = Field(default_factory=list, max_length=30)
    contradictions: list[Contradiction] = Field(default_factory=list, max_length=30)
    followups: list[SubQuestion] = Field(default_factory=list, max_length=100)
    coverage: list[Coverage] = Field(default_factory=list, max_length=50)


class Evidence(BaseModel):
    source_id: int = Field(gt=0)
    quote: str = Field(min_length=12, max_length=2000, description="Exact supporting passage copied from fetched page text")


class Fact(BaseModel):
    claim: str = Field(max_length=2000, description="One specific, verifiable statement")
    source_ids: list[int] = Field(max_length=16, description="[n] ids from search/fetch results that support the claim")
    evidence: list[Evidence] = Field(default_factory=list, max_length=16, description="Supporting quotes from fetched pages, one per source")


class RejectedFact(BaseModel):
    fact: Fact
    reason: str


class Findings(BaseModel):
    summary: str = Field(description="2-5 sentence answer to the sub-question")
    facts: list[Fact] = Field(default_factory=list, max_length=40)
    gaps: list[str] = Field(default_factory=list, description="Unanswered questions only; never speculative answers or omitted claims")
    rejected: list[RejectedFact] = Field(default_factory=list, description="Application diagnostics; leave empty")


class SynthesisReview(BaseModel):
    report_md: str = Field(min_length=1, description="Complete corrected report; preserve supporting numeric citations")
    issues: list[str] = Field(default_factory=list, max_length=30, description="Errors or unsupported assertions corrected or removed")


class SubQuestionResult(BaseModel):
    sub_question: SubQuestion
    round: int
    findings: Findings | None = None
    error: str | None = None
    searches: int = 0


class Source(BaseModel):
    id: int
    url: str
    title: str = ""
    published: str | None = None
    retrieved_at: str | None = None
    fetched: bool = False
    content_hash: str | None = None
    excerpts: list[str] = Field(default_factory=list)


class Stats(BaseModel):
    seconds: float = 0.0
    rounds: int = 0
    sub_questions: int = 0
    tool_calls: int = 0
    sources: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    fetches: int = 0
    model_calls: int = 0
    usage_incomplete: bool = False


class Report(BaseModel):
    question: str
    plan: list[SubQuestion]
    results: list[SubQuestionResult]
    report_md: str
    sources: list[Source]
    stats: Stats
    run_id: str = ""
    status: Literal["complete", "partial", "failed", "cancelled"] = "complete"
    warnings: list[str] = Field(default_factory=list)
    assessment: Assessment | None = None
    review: SynthesisReview | None = None
    assumptions: list[str] = Field(default_factory=list)
    clarifications: str = ""


class Answer(BaseModel):
    question: str
    answer_md: str
    sources: list[Source] = Field(default_factory=list)
    stats: Stats = Field(default_factory=Stats)
    run_id: str = ""
    status: Literal["complete", "partial", "failed"] = "complete"
    warnings: list[str] = Field(default_factory=list)
    review: SynthesisReview | None = None
