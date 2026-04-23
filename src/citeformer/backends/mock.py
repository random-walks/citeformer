"""Scripted backend used in unit tests (populated in P1).

`MockBackend` emits predetermined text given a prompt + sources. Lets us test the
orchestration layer and the grammar-building logic without loading a real model.
"""

from __future__ import annotations
