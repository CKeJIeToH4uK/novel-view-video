# Модельные интеграции

Папка содержит конкретные адаптеры внешних моделей. Она не является общим
framework: новый пакет появляется только вместе с реальным production-вызовом.

- `gen3c/` — лёгкие запросы и конкретные capability Gen3C/Cache4D.
- `grounded_sam2/` — один составной raw worker Grounding DINO boxes → SAM2
  instance masks → union boolean mask с временным RGB/mask протоколом.
- `dinov2/` — in-process ViT-B/14 owner: exact checkpoint load, bicubic
  462x840/ImageNet preprocess и raw 33x60 patch tokens.
- `lpips/` — in-process LPIPS 0.1 AlexNet owner: `TORCH_HOME`, точный
  spatial/eval constructor и forward из `[0,1]` в `[-1,1]` с
  `normalize=False`.
- `moge/` — один raw owner MoGe-v1: standalone NPY/backend в закреплённом
  Gen3C-окружении и прямой resident-вызов Cosmos helper внутри Gen3C.
- `vggt/` — три доказанные входные сетки, единый потоковый NPY-протокол,
  сырой результат, постоянный source-neutral record, явное camera-pack
  alignment и сборка metric posed depth VGGT-Omega, а также тяжёлый worker
  для EUVS и Waymo.

MoGe загружается только через `_inference.py`. Standalone worker делает один
load на последовательность и возвращает model-unit depth, mask и intrinsics;
Waymo metric scale остаётся у consumer. Resident Gen3C использует тот же
loader, но вызывает закреплённый Cosmos helper прямо в составном процессе:
нового subprocess или NPY-протокола у этого пути нет.

Grounded-SAM2 запускается fixed Gen3C Python без computation timeout;
полный лог пишет по явному постоянному пути, переданному consumer. Пакет
не знает EUVS-токенов, cache, PNG polarity или support: эти решения остаются
у evaluation consumer.

LPIPS не создаёт отдельного worker или NPY: один model instance живёт внутри
существующего evaluation composition process, а support-weighted reduction
остаётся у consumer.

DINOv2 также не создаёт transport: pinned read-only checkout присутствует в
`sys.path` только во время импорта factory, а cosine и fractional support
остаются у evaluation consumer.

Пакета `da3/` намеренно нет: экспериментальный DA3 находится в
`diffusion/code/research/da3_nested`, не входит в wheel и не имеет
поддерживаемого production-потребителя.

Верхнеуровневый `__init__.py` остаётся холодным и ничего не реэкспортирует.
