# Научные рецепты

Папка является read-only config mount root для научных параметров без
машинных путей, выбора данных и process topology.

- `ddw-preparation/` закрепляет научные и модельные параметры подготовки
  Waymo/DDW;
- `euvs-generation/` фиксирует VGGT source geometry и две политики Gen3C;
- `evaluation/` содержит mask recipe EUVS evaluation v1;
- `gaussian-generation/` разделяет пять существующих порядков v1–v5;
- `gen3c-training/` разделяет неизменный R4c LoRA baseline и
  экспериментальный LiDAR-depth recipe;
- `ddw-evaluation/` фиксирует matched base/v1/v2 heldout-point validation
  на явных checkpoint records одного фактического шага;
- `legacy/waymo/` хранит MoGe/VGGT checkpoints и разные depth gate v1/v2
  без путей машины и provenance-гейтов;
- внутренний lifecycle smoke по-прежнему использует inline-parameters.

Новая тематическая подпапка появляется только вместе с первым реальным job
и owner-specific разбором её параметров.
