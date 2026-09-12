# R4c LoRA с LiDAR-depth

## Гипотеза

Экспериментальный depth-loss может улучшить геометрическую согласованность
Gen3C относительно неизменного R4c LoRA baseline, не меняя способ обучения
LoRA и не дообучая depth observer вместе с моделью.

## Варианты

- baseline: [`jobs/waymo/r4c-lora-edm`](../../jobs/waymo/r4c-lora-edm/);
- эксперимент: [`jobs/waymo/r4c-lora-lidar-depth`](../../jobs/waymo/r4c-lora-lidar-depth/).

Обе job используют один PreparedRecord, один ordered split, один base model,
одинаковые training/validation seeds, число эпох и LoRA-конфигурацию. V2
сначала подгоняет observer только на clean training latents, проверяет сигнал
на validation samples, замораживает его и добавляет masked FRONT-A LiDAR
составляющую к прежнему EDM loss.

## Правило сравнения

Base, v1 и v2 сравниваются только на одном заранее выбранном фактическом
`completed_step`. Best-checkpoint каждой версии не используется для этого
A/B. Метрики считаются на heldout LiDAR points тех же восьми validation
segments, поэтому результат называется heldout-point validation, а не test.

## Статус

Технический CPU/fake контур подготовлен. Научный результат, CP4-поведение и
качество остаются `pending` до единой разрешённой кампании Stage 11 на
4×A100. До неё этот документ не утверждает, что v2 лучше baseline.
