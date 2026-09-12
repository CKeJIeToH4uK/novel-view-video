"""Shared error for the composite Grounded-SAM2 model boundary."""


class GroundedSam2ModelError(RuntimeError):
    """One raw Grounded-SAM2 request could not complete."""


__all__ = ["GroundedSam2ModelError"]
