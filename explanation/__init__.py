"""
I5 build block — Grounded Explanation + Diagnostic Questions.

A leaf, display-only layer consuming confirmation.schemas.RecomputeDiagnostic
(the same object ui/diagnostic_panel.py already renders from) and producing
a narrative explanation plus 3-5 ranked diagnostic questions. Never writes
back into EvaluationResult, confirmation state, or any governed assessment
object — see explanation/pipeline.py's module docstring for the full
architectural contract.

Nothing in domain/, engine/, confirmation/, or extraction/ is modified or
imported for write access by this package -- read-only consumption of
already-computed governed output only.
"""
