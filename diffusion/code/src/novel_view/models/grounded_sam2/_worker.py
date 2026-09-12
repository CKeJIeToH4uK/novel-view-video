"""Private composite Grounding DINO and SAM2 CUDA worker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--grounding-model-directory", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-model-config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--box-threshold", type=float, required=True)
    parser.add_argument("--text-threshold", type=float, required=True)
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    """Load both models once and segment each native frame independently."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rgb = np.load(
        arguments.exchange / "native_rgb.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    processor = AutoProcessor.from_pretrained(
        arguments.grounding_model_directory,
        local_files_only=True,
    )
    grounding_model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            arguments.grounding_model_directory,
            local_files_only=True,
            use_safetensors=True,
        )
        .to("cuda")
        .eval()
    )
    sam2_model = build_sam2(
        arguments.sam2_model_config,
        str(arguments.sam2_checkpoint),
        device="cuda",
        mode="eval",
        apply_postprocessing=True,
    ).eval()
    predictor = SAM2ImagePredictor(sam2_model)

    frame_count, height, width, _ = rgb.shape
    output = np.lib.format.open_memmap(
        arguments.exchange / "dynamic_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(frame_count, height, width),
    )
    try:
        for index in range(frame_count):
            image = Image.fromarray(
                np.array(rgb[index], dtype=np.uint8, copy=True, order="C"),
                mode="RGB",
            )
            inputs = processor(
                images=image,
                text=arguments.prompt,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                detections = grounding_model(**inputs)
            result = processor.post_process_grounded_object_detection(
                detections,
                input_ids=inputs.input_ids,
                threshold=arguments.box_threshold,
                text_threshold=arguments.text_threshold,
                target_sizes=[(height, width)],
            )[0]
            boxes = result["boxes"].detach().float().cpu().numpy()
            if boxes.shape[0] == 0:
                output[index] = combine_object_masks(
                    np.empty((0, height, width), dtype=np.bool_),
                    (height, width),
                )
                continue
            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                predictor.set_image(image)
                object_masks, _, _ = predictor.predict(
                    box=boxes,
                    multimask_output=False,
                    return_logits=False,
                )
            output[index] = combine_object_masks(
                object_masks,
                (height, width),
            )
        output.flush()
    finally:
        output._mmap.close()


def combine_object_masks(
    value: object,
    native_size_hw: tuple[int, int],
) -> np.ndarray:
    """Normalize SAM2's singleton axis and OR all instance masks."""
    masks = np.asarray(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[np.newaxis]
    if masks.shape[0] == 0:
        return np.zeros(native_size_hw, dtype=np.bool_)
    return np.logical_or.reduce(np.asarray(masks, dtype=np.bool_), axis=0)


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()
