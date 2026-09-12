"""Command-line boundary of :mod:`novel_view`.

Owns the lightweight internal job selector, planner, runner command, synthetic
worker command, build-candidate boundary and read-only attempt-record output
used by ``./distil3d``. Module :mod:`.legacy` only points to that launcher.
This package re-exports nothing, so importing it loads neither legacy commands
nor scientific modules.
"""
