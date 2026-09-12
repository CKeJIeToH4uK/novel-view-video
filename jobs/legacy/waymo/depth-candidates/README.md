# Сравнение глубины Waymo

`run.yaml` запускает один исторический трёхкамерный клип в порядке MoGe →
VGGT. Конкретный клип задаётся личным файлом
`selections/local/waymo-depth-clip.yaml`; образец лежит в
`selections/legacy/waymo/example-depth-clip.yaml`.

Задание сохраняет отдельный исход каждого метода и общий `result.json`.
Новые варианты научного порядка получают отдельный workflow, а не флаги в
этом задании.
