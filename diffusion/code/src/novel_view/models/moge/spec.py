"""Pinned facts shared by the two supported MoGe-v1 callsites."""


NUM_TOKENS = 2_500


class MogeError(RuntimeError):
    """One raw MoGe-v1 invocation could not complete."""


__all__ = ["MogeError", "NUM_TOKENS"]
