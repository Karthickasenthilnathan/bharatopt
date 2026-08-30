# bharatopt/solver/__init__.py
"""Solver package exposing the bounded‑variable dual simplex implementation."""

from .simplex import dual_simplex

__all__ = ["dual_simplex"]
