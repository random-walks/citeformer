"""Verification module for citeformer.

Populated in P6. The `VerificationReport` schema is locked in P1 so the §10.3
contract snapshot tests have something to pin; actual entailment / coverage /
existence logic lands with the NLI model integration in P6.
"""

from __future__ import annotations

from citeformer.verify.report import CitationSupport, VerificationReport

__all__ = ["CitationSupport", "VerificationReport"]
