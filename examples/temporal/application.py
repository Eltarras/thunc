"""Registered tasks for the Temporal example. Imports make no model calls."""

import os
from pathlib import Path

import thunc

workspace = Path(os.environ.get("THUNC_WORKSPACE", "./temporal-workspace")).resolve()
workspace.mkdir(parents=True, exist_ok=True)
agent = thunc.Agent("durable-reviewer", workdir=workspace, permissions=["!memory"], backend="openai")


@thunc.function(backend="openai")
def classify(question: str) -> str:
    """Classify the question in a short phrase."""
    ...


@agent.task
def analyze(question: str) -> str:
    """Read the workspace and answer the question. Cite the relevant files."""
    ...


@thunc.function(backend="openai")
def summarize(analysis: str) -> str:
    """Summarize the analysis in three short bullets."""
    ...
