"""Pipeline that turns the three upstream HF datasets into the warehouse parquets.

Public surface:
  - `pipeline.run(settings, ...)`: orchestrator entry point.
  - `pipeline.run_pipeline(...)`: same, used by CLI.
"""
