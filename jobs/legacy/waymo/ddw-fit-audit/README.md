# Визуальный аудит одного отказа

`run.yaml` запускает `waymo_ddw_audit/v1` через `./distil3d`. Вход — явно
названный rejected item с пройденным headroom и полным observed/reference.
Отдельный исторический AUDIT_MASK_GATE применяется до запуска моделей.

Затем тот же clip и вариант повторно рендерятся и сравниваются с сохранённым
измерением. В текущей attempt остаются condition RGB/known, preview.mp4 и
audit.json. Исходный item и accepted set не изменяются. Нужны его raw
Waymo Parquet и MoGe weights, но не Gen3C generation checkpoint.

Прежний rejected `clip/v1` распознаётся по собственной версии и читается
отдельным читателем. У него нет payload: старые campaign-relative пути
accepted items не переинтерпретируются как новые и этим аудитом не читаются.
Реальные locators храните в `jobs/local/`; новый научный порядок требует
отдельной версии.
