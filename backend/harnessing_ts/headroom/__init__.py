"""Residual-headroom core package.

Deterministic numpy/scipy implementation of the residual-headroom
method: synthetic tasks with exactly-known optimal forecasters, residual
diagnostics, per-coordinate linear correctors with held-out realized
gains, epsilon-saturation certification, and feedback rendering.

This package must not depend on the DSH harness, the orchestrator, or
any LLM.  Experiment drivers live under ``experiments/``.
"""
