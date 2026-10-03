"""String prompts: instructions built in code from data.

Rule of thumb: your own text (config, rubric) may go in the instructions; user content goes in
the inputs, never into the instructions through an f-string.

Run from the repo root:  python3 -m examples.dynamic_prompts
"""

import os
from dataclasses import dataclass

import thunc

thunc.configure(backend=os.environ.get("THUNC_BACKEND", "claude-code"))

# 1. thunc.call: one prompt per target language, assembled from a style guide.

STYLE_GUIDES = {
    "Dutch": "Use informal 'je', not 'u'. Keep product names in English.",
    "German": "Use formal 'Sie'. Keep product names in English.",
    "French": "Use 'vous'. Put a non-breaking space before ! and ?.",
}
release_note = "New: export your dashboards to PDF! Find it under Share → Export."

for language, rules in STYLE_GUIDES.items():
    translated = thunc.call(
        f"Translate the release note into {language}. Style rules: {rules}", {"release_note": release_note}
    )
    print(f"{language:7} {translated}")


# 2. @thunc.function(instructions=...): a typed, reusable function whose prompt comes from data.


@dataclass
class Grade:
    score: int
    feedback: str


QUESTION = "Why does HTTPS protect against eavesdropping?"
CRITERIA = [
    "mentions encryption of the traffic",
    "mentions that the server's identity is verified with a certificate",
    "is at most three sentences",
]
rubric = "\n".join(f"- {c}" for c in CRITERIA)


@thunc.function(
    instructions=f"Grade the student's answer to: {QUESTION}\nAward one point per criterion met:\n{rubric}\n"
    "Give one sentence of feedback naming what is missing, if anything.",
    ensure=lambda g: 0 <= g.score <= len(CRITERIA),
)
def grade(answer: str) -> Grade: ...


answers = [
    "Because the data is encrypted so nobody in between can read it.",
    "TLS encrypts traffic, and the certificate proves you're talking to the real server.",
]
for answer, result in zip(answers, thunc.map(grade, answers), strict=True):
    print(f"\n{result.score}/{len(CRITERIA)}  {answer}\n     {result.feedback}")
