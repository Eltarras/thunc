"""thunc (think + function): call an LLM like a typed Python function.

import thunc
thunc.configure(backend="claude-code")

# A docstring, for prompts that should read like code:
@thunc.function
def urgency(ticket: str) -> int:
    \"\"\"Rate how urgent this ticket is, from 1 to 5.\"\"\"
    ...

# A string, for prompts built in code:
thunc.call(f"Translate into {language}.", {"text": note})
"""

from . import prompts
from .agent import Agent, agent
from .cache import CacheGroup, cache_info, clear_cache
from .config import configure
from .core import call, map
from .decorator import function
from .errors import ThuncError
from .runs import AgentError, Run

__all__ = [
    "Agent",
    "AgentError",
    "CacheGroup",
    "ThuncError",
    "cache_info",
    "call",
    "clear_cache",
    "configure",
    "function",
    "map",
    "prompts",
    "Run",
    "agent",
]
__version__ = "0.2.1"
