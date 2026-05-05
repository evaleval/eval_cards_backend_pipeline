"""Pure interpretive-signal logic. I/O-free; safe to unit test in isolation.

Modules:
- reproducibility: per-row reproducibility-gap rule + active field set.
- comparability: variant-divergence + cross-party-divergence over a group.
- completeness: per-fact-row reporting completeness over the operationalised field set.
- setup: setup-field normalisation + canonical-JSON helpers (variant_key + differing-field source).

Provenance has no Python module here. The per-row source-type collapse is
inline in `canonicalise/stages.py` Stage E (one COALESCE expression). The
group-derived multi-source / first-party-only fields are computed in
`canonicalise/stages.py` Stage F.1 via SQL window functions / CTEs — pure
SQL is the right tool for that group-and-broadcast shape.
"""
