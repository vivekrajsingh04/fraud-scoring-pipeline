"""Test package.

Present so ``tests.conftest`` has exactly one module name. Without it mypy sees
the same file as both ``conftest`` and ``tests.conftest`` and refuses to run,
and ``from tests.conftest import ...`` in the parity test would depend on the
working directory.
"""
