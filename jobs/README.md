# Задачи запуска

Каждая подпапка job соединяет один workflow, вход, recipe либо inline-
parameters и execution preset в `run.yaml`. Машинные пути сюда не входят.

- `examples/` содержит малые полностью отслеживаемые примеры;
- `acceptance/` содержит 11 уникальных EUVS/Gaussian форм Stage 11 и
  обезличенный список внешних входов; run-specific копии остаются в
  `local/`;
- `waymo/` содержит переносимые задания подготовки Waymo/DDW и R4c training;
- `legacy/` связывает сохранённые EUVS scopes, Scene9 и исторические Waymo
  keysets, depth selection/comparison, DDW canary/probe/survey и fit/audit
  с уже работающими версиями workflow; точные данные и попытки
  задаёт оператор в личной копии;
- будущие предметные группы добавляются вместе с работающим workflow;
- личные задания хранятся в полностью игнорируемой `local/`; туда же
  помещаются evaluation jobs с явными путями к конкретным prepared и
  checkpoint records.

Новый job добавляется только вместе с командой `./distil3d plan`, которая
может его полностью разрешить.
