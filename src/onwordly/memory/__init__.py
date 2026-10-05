"""Add-only, non-parametric memory: episode store, variant register, trust ledger.

Pure Python (no ML dependencies). See docs/model-design.md.
"""
from onwordly.memory.register import (
    Divergence,
    RegisterSummary,
    VariantEntry,
    VariantRegister,
    arithmetic_frame,
    canonical_filler,
    frame_for,
    rules_frame,
)
from onwordly.memory.scaling import UPDATE_RULES, normalize_weights, update_weight
from onwordly.memory.store import GENESIS_HASH, RECORD_KINDS, EpisodeStore, Record, verify_chain
from onwordly.memory.trust import BetaTrust, TrustLedger

__all__ = [
    "BetaTrust",
    "Divergence",
    "EpisodeStore",
    "GENESIS_HASH",
    "RECORD_KINDS",
    "Record",
    "RegisterSummary",
    "TrustLedger",
    "UPDATE_RULES",
    "VariantEntry",
    "VariantRegister",
    "arithmetic_frame",
    "canonical_filler",
    "frame_for",
    "normalize_weights",
    "rules_frame",
    "update_weight",
    "verify_chain",
]
