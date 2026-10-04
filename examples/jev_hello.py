"""The smallest Jev calls: a yes/no, a label and a rating.

Jev answers in well under a second, but only bool and Literal[...] return types.
Needs the jev CLI, logged in (see docs/jev.html).  Run from the repo root:  python3 -m examples.jev_hello
"""

from typing import Literal

import thunc

thunc.configure(backend="jev")

email = "You won a free iPhone! Click here to claim it."
review = "Arrived late but works great."
ticket = "Nobody on our team can log in since this morning."

spam = thunc.call("Is this email spam?", {"email": email}, returns=bool)
mood = thunc.call(
    "What is the mood of this review?", {"review": review}, returns=Literal["positive", "mixed", "negative"]
)
level = thunc.call(
    "How urgent is this, from 1 (can wait) to 5 (blocked)?", {"ticket": ticket}, returns=Literal[1, 2, 3, 4, 5]
)

print(f"spam: {spam}   mood: {mood}   urgency: {level}")
