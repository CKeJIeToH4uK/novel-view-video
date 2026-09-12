# ddw-probe

Исторический `waymo_ddw_probe/v1`. Перед запуском заполните
`selections/local/waymo-ddw-canary.yaml` по форме
`selections/legacy/waymo/example-depth-clip.yaml` и укажите точный
относительный путь готового `depth-selection.json` в поле `input.depth_selection`.
Пример пути в job не выбирает последний результат автоматически.

Модель и данные подключаются через profile; образ — `moge`, одна GPU.
Здесь хранится только форма запуска. Настоящая аппаратная проверка — Stage 11.
