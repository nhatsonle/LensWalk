"""Compatibility exports for prompt constants.

New code should import prompt templates from :mod:`vcs.prompts`. This module is
kept so existing scripts that reference ``vcs.constants`` continue to run.
"""

from .prompts import *  # noqa: F401,F403
