"""Torch-free record factories shared by Gen3C checkpoint/workflow tests."""

from novel_view.training.gen3c.lora.depth_record import CheckpointRecordV2
from novel_view.training.gen3c.lora.depth_spec import (
    DepthObjectiveV2,
    DepthObserverFitIdentity,
    DepthObserverFitResult,
    DepthObserverFitSpec,
)
from novel_view.training.gen3c.lora.record import (
    AbTrainingIdentityV1,
    CheckpointRecordV1,
)


def training_identity(*, seed: int = 7) -> AbTrainingIdentityV1:
    return AbTrainingIdentityV1(
        "gen3c/official/model.pt",
        "waymo-ddw/r4c-ddw-prepared/prepared.json",
        ("train-a", "train-b"),
        ("segment-a", "segment-b"),
        ("validation-a",),
        ("segment-v",),
        seed,
        1,
        2,
    )


def depth_facts(identity: AbTrainingIdentityV1 | None = None):
    fit = DepthObserverFitSpec(
        0, 0, 10, 1, "adamw", 0.001, (0.9, 0.999), 1e-8, 0.0, "constant"
    )
    split = (
        (
            ("sample-train-a", "sample-train-b"),
            ("segment-train-a", "segment-train-b"),
            ("sample-validation",),
            ("segment-validation",),
        )
        if identity is None
        else (
            identity.training_sample_ids,
            identity.training_segment_ids,
            identity.validation_sample_ids,
            identity.validation_segment_ids,
        )
    )
    return (
        DepthObjectiveV2(0.1),
        DepthObserverFitIdentity(*split, fit),
        DepthObserverFitResult(0.03, 0.04, 0.06),
    )


def checkpoint_record(step: int, loss: float, attempt: str, version: int):
    identity = training_identity(seed=0)
    if version == 1:
        return CheckpointRecordV1(
            identity,
            step,
            loss,
            attempt,
            f"step-{step:09d}.pt",
        )
    objective, fit_identity, fit_result = depth_facts(identity)
    return CheckpointRecordV2(
        objective,
        identity,
        fit_identity,
        fit_result,
        "train/run/fresh",
        step,
        loss - 0.01,
        0.1,
        loss,
        attempt,
        f"step-{step:09d}.pt",
    )
