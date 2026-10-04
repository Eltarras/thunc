from typing import Literal

import thunc


@thunc.function
def route(ticket: str) -> Literal["bug", "billing", "feature request"]:
    """Which team should handle this support ticket?"""
    ...


@thunc.function
def urgency(ticket: str) -> Literal[1, 2, 3, 4, 5]:
    """Rate how urgent this ticket is, from 1 (can wait) to 5 (customer is blocked)."""
    ...


def test_choice():
    assert route("I was charged twice for my subscription this month.") == "billing"
    assert route("The app crashes every time I open the settings page.") == "bug"


def test_score_ranks_a_blocker_above_a_wish():
    blocked = urgency("Production is down and none of our customers can log in.")
    wish = urgency("It would be nice to have a dark mode some day.")
    assert blocked >= 4 and wish <= 2
