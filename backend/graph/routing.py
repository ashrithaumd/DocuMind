"""Small-talk detection for the routing node.

Deliberately a heuristic, not an LLM call. An LLM classifier would add a
model call to EVERY real query to catch cases the pipeline already handles
safely — an off-topic question ("what's the weather?") goes through grading
and gets a clean refused_no_context. The heuristic only needs to catch the
obvious conversational messages where running retrieval at all is silly.

The costs of a mistake are asymmetric, and the patterns are shaped for that:
 - false negative (greeting slips through): one wasted cheap pipeline run
   that ends in a polite refusal — mildly inefficient, never wrong.
 - false positive (real question canned): actively bad, so every pattern is
   anchored to match the WHOLE message. "hello, what does section 3 say?"
   does not match; bare "hello!" does.

Being a pure function of the input string, it is also unit-testable and
replayable, which fits this project's fixed-pipeline discipline.
"""
import re

_SMALL_TALK = re.compile(
    r"""^(?:
        (?:hi+|hello+|hey+|heya|yo|howdy|greetings|good\s+(?:morning|afternoon|evening))
      | (?:thanks|thank\s+you|thx|ty|cheers)(?:\s+(?:so\s+much|a\s+lot|very\s+much))?
      | (?:how\s+are\s+you(?:\s+doing)?|how'?s\s+it\s+going|what'?s\s+up|sup)
      | (?:bye|goodbye|good\s+night|see\s+you|see\s+ya|later|take\s+care)
      | (?:ok|okay|cool|nice|great|awesome|perfect|got\s+it|sounds\s+good)
      | (?:who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help)
      | (?:test|testing|ping)
    )[\s!.?,:;)…]*$""",
    re.IGNORECASE | re.VERBOSE,
)

SMALL_TALK_RESPONSE = (
    "Hi! I'm DocuMind. Upload a PDF, DOCX, or TXT document and ask me "
    "questions about it — answers cite the document, and if it doesn't "
    "contain the answer, I'll say so instead of guessing."
)


def is_small_talk(question: str) -> bool:
    return bool(_SMALL_TALK.match(question.strip()))
