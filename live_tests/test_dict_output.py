"""Raw, messy text in; a dict out. The log mixes decimal commas (42,50), "€134,-" for 134.00,
a "2x ... each" multiplication, and several date and currency spellings."""

import pytest

import thunc

RAW_EXPENSES = """\
03/09 - Uber airport -> hotel  €42,50 (Sara)
lunch @ Bistro Amsterdam: 18.90 EUR - Tom
2026-09-03 coffee x3 7.20
Sept 4th, train tickets AMS-RTM 2x €16.80 each
hotel 2 nights 289.00 eur  paid by company card
4/9 dinner w/ client  €134,-
taxi back 38 euro (receipt lost)
coffee 2,40
Train RTM-AMS €33.60 total
04-09 snacks for the booth: €12.35
"""

# transport: 42.50 + 2 x 16.80 + 38.00 + 33.60
# food:      18.90 + 7.20 + 134.00 + 2.40 + 12.35
# lodging:   289.00
EXPECTED = {"transport": 147.70, "food": 174.85, "lodging": 289.00}


@thunc.function(ensure=lambda d: set(d) == {"transport", "food", "lodging"})
def totals_by_category(expenses: str) -> dict[str, float]:
    """Total these trip expenses per category: transport, food and lodging. All amounts are in euros."""
    ...


def test_totals_by_category():
    totals = totals_by_category(RAW_EXPENSES)
    assert totals == pytest.approx(EXPECTED, abs=0.01)
