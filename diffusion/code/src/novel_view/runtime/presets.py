"""Доказанные процессные пресеты нового launcher."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionPreset:
    name: str
    gpu_count: int


CPU_TEST = ExecutionPreset(name="cpu_test", gpu_count=0)
INFERENCE_CP1 = ExecutionPreset(name="inference_cp1", gpu_count=1)
INFERENCE_CP2 = ExecutionPreset(name="inference_cp2", gpu_count=2)
TRAINING_CP4 = ExecutionPreset(name="training_cp4", gpu_count=4)


def load_execution_preset(name: str) -> ExecutionPreset:
    if name == CPU_TEST.name:
        return CPU_TEST
    if name == INFERENCE_CP1.name:
        return INFERENCE_CP1
    if name == INFERENCE_CP2.name:
        return INFERENCE_CP2
    if name == TRAINING_CP4.name:
        return TRAINING_CP4
    raise ValueError(f"unsupported execution preset: {name}")
