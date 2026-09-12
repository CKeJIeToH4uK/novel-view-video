# Lifecycle smoke

Внутренняя синтетическая job проверяет общий путь `plan` и Python-часть
attempt lifecycle: runner, отдельную process group, worker и `attempt.json`.
Она не запускает научный код, GPU, данные или модели и не принимает
произвольную команду из YAML. Host `run/resume`, foreground/detach и
per-attempt container уже используют этот workflow. `status/logs` покрывают
его record/Docker-состояния; `stop` проверяет мягкую и принудительную остановку
всей process group, а `remove` удаляет только завершённый контейнер.

Текущий `run.yaml` выбирает успешный режим с кодом `0` и preset `cpu_test`.
