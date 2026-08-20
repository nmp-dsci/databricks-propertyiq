"""The judged eval's question set — 6 broad + 6 narrow, all corpus-grounded.

Stratification is the point (plan s09, decision D2): broad questions are the
multi-hop targets the optimisation matrix must improve; narrow questions are
the no-regression controls where a single retrieval is often genuinely
sufficient and staying single-hop is *correct*. The gate reads the two strata
separately, so a variant cannot buy broad-question depth by thrashing narrow
questions with pointless extra hops.

Provenance: narrow questions and three of the broad ones reuse
`silver_golden_qa` cases (transcript-lab's retrieval goldens — already
verified against the corpus), keeping this set consistent with the retrieval
benchmark. The remaining broad questions are grounded in the corpus themes
table and a coverage probe: 162 chunks across 14 videos mention Databricks,
including three solutions-architect interview videos, which is what makes the
canonical SA-interview question answerable at depth.
"""

from __future__ import annotations

BROAD = "broad"
NARROW = "narrow"

# The user's own demo question, kept close to their words. This is the case
# the whole branch exists to fix: dev research-loops it in 5-8 retrievals,
# the Databricks agentic mode answered it from one.
CANONICAL_QUESTION_ID = "b01"

QUESTIONS: list[dict[str, str | None]] = [
    # -- broad: synthesis across many videos; the multi-hop targets ----------
    {
        "question_id": "b01",
        "category": BROAD,
        "domain": "databricks",
        "question": (
            "Can you give me a guide into what I need to know about Databricks? "
            "I'm interviewing for a solutions architect role."
        ),
        "source_case_id": None,
    },
    {
        "question_id": "b02",
        "category": BROAD,
        "domain": "ai-coding",
        "question": (
            "According to this corpus, how do senior engineers structure their "
            "agentic coding setups?"
        ),
        "source_case_id": "g012",
    },
    {
        "question_id": "b03",
        "category": BROAD,
        "domain": "property",
        "question": (
            "What are the recurring arguments the property videos make about the "
            "federal budget's tax changes?"
        ),
        "source_case_id": "g011",
    },
    {
        "question_id": "b04",
        "category": BROAD,
        "domain": "career",
        "question": (
            "What advice do the job-search videos in this corpus repeat across "
            "resumes, LinkedIn and interviews?"
        ),
        "source_case_id": "g020",
    },
    {
        "question_id": "b05",
        "category": BROAD,
        "domain": "system-design",
        "question": (
            "What do the system design videos teach about consistency, latency "
            "and the trade-offs between them in distributed systems?"
        ),
        "source_case_id": None,
    },
    {
        "question_id": "b06",
        "category": BROAD,
        "domain": "ai-coding",
        "question": (
            "How is AI changing what software engineers are actually valued for, "
            "according to these videos?"
        ),
        "source_case_id": None,
    },
    # -- narrow: one video usually answers them; the no-regression controls --
    {
        "question_id": "n01",
        "category": NARROW,
        "domain": "property",
        "question": (
            "If I bought an investment property before budget night, do I keep negative gearing?"
        ),
        "source_case_id": "g002",
    },
    {
        "question_id": "n02",
        "category": NARROW,
        "domain": "ai-coding",
        "question": "How do I set up Herder, and what app do I run it in?",
        "source_case_id": "g008",
    },
    {
        "question_id": "n03",
        "category": NARROW,
        "domain": "career",
        "question": "How do I make my resume ATS-friendly?",
        "source_case_id": "g015",
    },
    {
        "question_id": "n04",
        "category": NARROW,
        "domain": "property",
        "question": "Is the Gold Coast property market at risk of collapse, and why?",
        "source_case_id": "g004",
    },
    {
        "question_id": "n05",
        "category": NARROW,
        "domain": "career",
        "question": "What should the professional summary at the top of a resume say?",
        "source_case_id": "g016",
    },
    {
        "question_id": "n06",
        "category": NARROW,
        "domain": "ai-coding",
        "question": "How should I use memory files and skills to onboard a coding agent?",
        "source_case_id": "g009",
    },
]


def by_id(question_id: str) -> dict[str, str | None]:
    for entry in QUESTIONS:
        if entry["question_id"] == question_id:
            return entry
    raise KeyError(question_id)
