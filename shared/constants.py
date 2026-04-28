"""Constants used across the application."""

from enum import Enum


class EvalStatus(str, Enum):
    """Status of an evaluation bundle in the pipeline."""

    UPLOADING = "UPLOADING"
    VALIDATING = "VALIDATING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
