# Быстрый запуск diffusion

Все команды выполняются из корня репозитория на Linux/amd64 с Docker.
Для модельных задач нужен настроенный NVIDIA runtime и выбранные GPU.
Хостовые Python, Conda и ручной `PYTHONPATH` не требуются.
Docker-сборка и контейнерные проверки рассчитаны на Linux.

Это инструкция запуска, не утверждение о завершённой A100-приёмке или
разрешение на чужие веса и данные. Сначала учитывайте
[условия компонентов](licensing.md); передача на A100 описана
[отдельно](a100-operator.md).

## 1. Профиль машины

Скопируйте пример и подставьте свои пять абсолютных путей и список GPU:

```bash
mkdir -p profiles/local/my-server
cp profiles/examples/a100-4gpu/.env.example profiles/local/my-server/.env
```

В `.env` изменяются `DISTIL3D_DATA_ROOT`, `DISTIL3D_MODELS_ROOT`,
`DISTIL3D_PREPARED_ROOT`, `DISTIL3D_RUNS_ROOT`, `DISTIL3D_CACHE_ROOT` и
`DISTIL3D_GPU_IDS`. Для одной карты, например, последний равен `0`.
Профиль находится вне Git. Значения не оборачиваются в кавычки;
подстановок команд и переменных в этом файле нет.

В контейнере эти каталоги всегда видны как `/data`, `/models`, `/prepared`,
`/runs`, `/cache`. Поэтому пути одного сервера не попадают в научные recipes.
Раскладка и грамматика подробнее в [профилях](../profiles/README.md).

## 2. Образ и план

```bash
./distil3d build
./distil3d plan --profile profiles/local/my-server jobs/acceptance/euvs-pair-autoregressive-v1
```

`build` собирает основной образ `core`, не принимает профиль и не запускает
модель. Для MoGe отдельно используется `./distil3d build --variant moge`.
Обычные команды используют готовые образы без скрытой сборки.

`plan` показывает разрешённую конфигурацию и будущую команду контейнера,
не открывает данные/веса и не создаёт попытку. Такой успех означает
«конфигурация разобрана», а не «данные и GPU готовы». Заметка VGGT, если
этот backend используется задачей, информационная и ничего не блокирует.

## 3. Настоящая задача

Tracked jobs/selections содержат переносимые формы, иногда синтетические
значения. Для запуска нужен конкретный набор входов, а для потребителя —
точная ссылка на результат предыдущего сценария.

```bash
mkdir -p jobs/local
cp -R jobs/acceptance/euvs-pair-autoregressive-v1 jobs/local/my-euvs
```

В копии задайте своё имя задачи, реальную selection и нужный source-views
attempt. Сами данные и веса положите во внешние каталоги профиля.
Не заменяйте отсутствующие результаты вымышленными путями.
Как устроены ссылки, описано в [jobs](../jobs/README.md),
[recipes](../recipes/README.md) и [selections](../selections/README.md).
Точные роли входов конкретных сценариев перечислены в
[EUVS/Gaussian/историческом Waymo](../jobs/acceptance/STAGE11_HANDOFF.md)
и [Waymo/R4c](../jobs/waymo/STAGE11_HANDOFF.md).

## 4. Диагностика — по желанию

```bash
./distil3d doctor host --profile profiles/local/my-server
./distil3d doctor job --profile profiles/local/my-server jobs/local/my-euvs
```

Первая проверяет машину и запись в prepared/runs/cache; data/models только
показывает. Вторая открывает нужные конкретной задаче ресурсы и проверяет
окружения; для GPU выполняет короткие аппаратные пробы. Это не запуск
самого опыта. Обычные `plan/run/resume` не вызывают doctor автоматически.

## 5. Запуск и результат

```bash
./distil3d run --profile profiles/local/my-server jobs/local/my-euvs
```

Команда печатает точную ссылку вида `job-name/run-id/attempt-id`.
Используйте именно её в следующих командах, заменив пример:

```bash
./distil3d status --profile profiles/local/my-server job-name/run-id/attempt-id
./distil3d logs --profile profiles/local/my-server job-name/run-id/attempt-id -f
```

Запись `attempt.json`, логи моделей и результаты находятся внутри
соответствующей попытки в каталоге runs профиля. `status` показывает
и запись Python, и фактическое состояние Docker.

Для отсоединённого запуска добавьте `--detach` к `run`. Ctrl-C при обычном
`run` прекращает только просмотр логов: контейнер продолжает работу.
Остановить его нужно явно:

```bash
./distil3d stop --profile profiles/local/my-server job-name/run-id/attempt-id
```

Для возобновления обучения передайте точную исходную попытку и выбранный
checkpoint по пути внутри контейнера:

```bash
./distil3d resume --profile profiles/local/my-server job-name/run-id/attempt-id --checkpoint /runs/path/to/checkpoint.pt
```

Нет поиска «последнего» checkpoint и дополнительной проверки совместимости:
его читает загрузчик выбранного метода. Поддержка resume зависит от workflow.

`remove` удаляет завершённый контейнер, сохраняя результаты и `attempt.json`.
Контейнерные логи при этом теряются — заберите нужный отчёт до удаления.
Полная справка: `./distil3d --help`. Обучение, evaluation и старые сценарии
используют тот же интерфейс jobs, без отдельных shell-запускателей.
