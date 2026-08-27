"""
Milestone 3B — Confirmation UX / Interaction Flow (design-approved
prototype: Streamlit, single process, in-memory only -- no auth, no
persistence, no deployment; see MANIFEST.txt for the full checkpoint).

Every module here is a thin adapter over the completed Milestone 3A
`confirmation/` backend. Nothing in this package reproduces confirmation-
state, active-evidence, consequentiality, or recomputation logic -- every
governed value rendered by the UI comes from a fresh
confirmation.recompute.recompute() call made in ui/actions.py.
"""
