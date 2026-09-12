# Исторические рецепты Waymo

- `depth-comparison.yaml` задаёт checkpoints двух сравниваемых методов;
- `depth-gate-v1.yaml` сохраняет исходный порог heldout-точек;
- `depth-gate-v2.yaml` сохраняет исправленный candidate-neutral gate.
- `ddw-full-axis.yaml` — прежняя ось d1..4 × оба знака для engineering
  canary и survey; их разные правила остановки определяются workflow;
- `ddw-canary-v2.yaml` — ось d2..4 × оба знака;
- `ddw-probe.yaml` — ось d3/4 × оба знака.
- `ddw-fit.yaml` — MoGe checkpoint для назначенного fit и visual audit;
  прежние разные fit/audit gates принадлежат соответствующим порядкам.

DDW-рецептам нужен только MoGe checkpoint. Cache4D reference и forward warp
используют установленный Gen3C-код для геометрии, без generation-весов.

Новые пороги или другой порядок сравнения добавляются новой версией, а не
скрытым флагом.
