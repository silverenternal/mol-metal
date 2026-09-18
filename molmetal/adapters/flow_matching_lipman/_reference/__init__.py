"""Stable import/pickle entry point for the isolated optional Meta reference.

The original source files remain unchanged; their absolute imports are handled
by the private loader. MolFlow's public ``flow_matching`` package stays intact.
"""
from .._reference_loader import configure_reference

__path__ = [str(configure_reference())]
