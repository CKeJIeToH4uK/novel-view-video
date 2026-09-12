# Исторические задания Waymo

Папка хранит явные задания прежних опытов Waymo. `keysets/` выбирает
центральные ключи до чтения RGB и геометрии; `depth-candidates/` выполняет
MoGe→VGGT для одного клипа; `depth-selection-v1/` и `depth-selection-v2/`
сохраняют два разных научных решения над восемью reports.

Отдельные формы `ddw-canary-v1/`, `ddw-canary-v2/`,
`ddw-probe/` и `ddw-survey/` для отдельных исторических порядков DDW.
Их вход — явный clip и готовый depth-selection record, а не индекс в
найденной на машине коллекции. Рецепты содержат только ось и модель;
`ddw-fit/` считает явно выбранные назначенные варианты,
`ddw-fit-collect/` собирает только перечисленные item records, а
`ddw-fit-audit/` визуализирует один eligible rejected item без изменения
accepted set. Эти три порядка не используют старую очередь машин.

Полные списки сегментов находятся в исключённом из Git `selections/local/`.
Исходный `waymo-segment-split/v1` сохранён без изменения в
`selections/local/waymo-segments-v1.yaml` — это точный вход `keysets/`.
Исторические fit/dev/debug используют official training; они не заменяют
новый R4c training/validation split. Каждая новая папка получает краткий
README, работающий workflow и только необходимые ссылки на входы.

## Передача исторических DDW-сценариев

Для финальной Stage 11 нужны: один прежний exposed-debug clip для canary v1,
прежний canary-v2 clip для v2/probe и явно выбранный survey clip. Их реальные
121 timestamps и segment ID записываются только в `selections/local/` по
форме `selections/legacy/waymo/example-depth-clip.yaml`; конкретные local
имена приведены в README каждой job. Старый закрытый v2-вход сохранён в
`selections/local/stage11/legacy/waymo_ddw_v2_preregistration.json`.

Узлу нужны raw Waymo Parquet этих clips, MoGe-ViT-L веса из recipe и
результат выбора глубины с выбранным MoGe. Gen3C checkpoint этим четырём
сценариям не нужен: reference использует только Cache4D-геометрию. После
depth-selection оператор явно указывает его runs-relative locator в job.
Это ещё не разрешение запускать модели или передавать данные наружу;
конкретное аппаратное окно остаётся Stage 11.

В единой `accept a100-4` шесть существующих форм — depth-candidates,
depth-selection-v2 и четыре canary/probe/survey — копируются в
`jobs/local/<campaign>/`. Приёмка подставляет входы из
`selections/local/stage11/legacy/` и точный результат своего CPU gate;
tracked jobs оператор не редактирует. Полный набор восьми входных candidate
reports передаётся отдельно и не заменяется одним новым comparison.
Точные роли, форматы и объём перечислены в
[`jobs/acceptance/STAGE11_HANDOFF.md`](../../acceptance/STAGE11_HANDOFF.md).
Обычные прямые запуски этих форм сохраняют описанные выше личные пути.

Fit/collection используют прежний полный keyset из `waymo_keysets/v1`:
варианты распределяются до выбора поднабора. Закрытая исходная assignment
сохранена для сопоставления в игнорируемом
`selections/local/stage11/legacy/waymo_ddw_fit_assignment_v1.json`;
новый runtime не требует ещё одного assignment manifest. Для fit/audit
достаточно raw Parquet нужных clips, MoGe weights и явных входов. Полные
718 items не пересчитываются ради приёмки переноса.
