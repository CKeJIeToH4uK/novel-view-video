# Исторические выборки Waymo

Файлы показывают внешние формы исторических Waymo-сценариев:

- `example-depth-clip.yaml` — один полностью определённый 121-frame clip;
- `example-depth-choice-8.yaml` — восемь ordered locators результатов под
  `/runs`.
- `example-fit-subset.yaml` — полный keyset и явные индексы для fit;
- `example-fit-collection.yaml` — полный keyset и явные item-record locators
  без поиска каталогов. Результат отмечает отсутствующие строки как incomplete.

Настоящие закрытые ключи и locator запусков копируются в одноимённые файлы
под `selections/local/`, которая не попадает в Git.

DDW canary v1 использует `selections/local/waymo-ddw-exposed-debug.yaml`;
canary v2 и probe — `waymo-ddw-canary.yaml`, survey —
`waymo-ddw-survey.yaml`. Все три файла имеют форму `example-depth-clip.yaml`.
Исторический canary-v2 clip можно восстановить из сохранённой локальной
`selections/local/stage11/legacy/waymo_ddw_v2_preregistration.json`.
Это ручная подготовка входа, не runtime discovery. Конкретный результат
depth-selection задаётся в job отдельно.
