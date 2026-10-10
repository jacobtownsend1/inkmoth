"""Public library API. Model integration is loaded only when a real run starts."""
from importlib.metadata import version

__version__ = version("inkmoth")  # first: submodules import it

from .config import PRESETS, Settings
from .context import RunContext
from .pipeline import run_ask_result, run_clarify, run_research
from .schemas import Answer, ClarificationCheck, Report

__all__ = ["Settings", "PRESETS", "RunContext", "Report", "Answer", "ClarificationCheck", "run_research",
           "run_clarify", "run_ask_result"]
