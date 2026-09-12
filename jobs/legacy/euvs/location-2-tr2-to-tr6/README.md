# EUVS location-2-tr2-to-tr6

Первая историческая отдельная пара: location 2, source traversal 2 → target traversal 6.
Порядок пар и source/target токенов задаёт
`selections/euvs/euvs-location-2-tr2-to-tr6.yaml`.

1. Выполнить `source-views/run.yaml` через корневой `./distil3d` с
   подходящим профилем: задание `euvs-location-2-tr2-to-tr6-source-views-v1`
   подготавливает геометрию исходных кадров этой выборки.
2. Подставить точные `replace-run` и `replace-attempt` завершённого
   source-views запуска в `input.source_views_attempt` выбранной формы.
3. Выбрать `run.yaml` для `euvs_generation/v1` с авторегрессией
   либо `source-reseed/run.yaml` для `euvs_generation/v2` с повторным
   началом каждого окна от исходных кадров. Обе формы ссылаются на тот же
   `euvs-location-2-tr2-to-tr6-source-views-v1/replace-run/attempts/replace-attempt`.
   Существующие рецепты задают seed 0 и 35 шагов; профиль исполнения —
   `inference_cp1`, набор данных — `euvs`.

Для оценки переиспользовать форму
`jobs/acceptance/euvs-evaluation-run-v1/run.yaml`: задать эту же
`input.selection` и точную ссылку `input.generation.path` на
`euvs-location-2-tr2-to-tr6-autoregressive-v1/<run>/attempts/<attempt>` либо
`euvs-location-2-tr2-to-tr6-source-reseed-v2/<run>/attempts/<attempt>`.
Для сравнения переиспользовать
`jobs/acceptance/euvs-base-tuned-comparison-v1/run.yaml` с той же выборкой
и точными ссылками `base_evaluation`/`tuned_evaluation`. Локальные
копии этих форм получают собственные имена заданий.

Primary35, extended16 и raw29 запускают, оценивают и публикуют раздельно.
Основная таблица относится только к primary35; raw29 не агрегируют ни с
основной метрикой, ни с extended16. Результат отдельной пары также не
подменяет результат полной выборки.

В папке хранятся только эти три задания одного исторического состава.
Другой состав получает отдельную выборку и папку заданий; общие параметры
остаются в рецептах.
