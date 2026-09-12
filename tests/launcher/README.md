# Проверки host launcher

`test_distil3d.py` запускает настоящий корневой Bash launcher, но подставляет
маленький fake Docker. Набор проверяет внешний разбор команд, profile и форму
Docker build/run без вложенного Docker Engine, GPU, данных и моделей. Для
`plan` он отдельно закрепляет два `--rm` контейнера, три read-only config
mount, отсутствие machine-root bind mounts и настоящий exit code финального
контейнера. Python-набор входит в `integration` и `all`; настоящий Docker
использует только следующий за `all` отдельный сценарий.

Production selection проверяется по фиксированному core tag, а final plan
и attempt — по exact selected image ID. Тесты исключают скрытый build,
смену tag и продолжение после ошибки окончательного плана.
Компактный набор сохраняет различающие случаи, которых нет в обычном
жизненном цикле: повторный resume без контейнера-источника, короткую ссылку
активной попытки, преждевременный конец потока логов при живом контейнере,
естественное завершение одновременно с stop и отдельный признак OOM.
Ресурсы doctor и attempt сравниваются с настоящим Python-планом.

`test_lifecycle.sh`, автоматически вызываемый группой
`./distil3d test all` после pytest, на настоящем Linux Docker проверяет
уже реализованные
success 0, failure 23, detach, Ctrl-C 130 без остановки контейнера и новый
resume-attempt с opaque checkpoint. Матрица наблюдаемости дополнительно
проверяет record-absent created/exited-137, running, succeeded, failed,
stale running после SIGKILL, removed container, выбор последней записи,
неоднозначность двух активных attempts, `logs`/`logs -f`, неблокирующий
`status` и неизменность хешей records. Fake Docker отдельно даёт
`OOMKilled=true`, поэтому code 137 сам по себе не называется OOM.
Он также проверяет явный SIGTERM для leader/child/grandchild, мягкое
`stopped/143`, принудительное `running/137`, исчезновение всех host PID,
запрет удаления running-container и сохранение attempt-root при повторном
`remove`. Сценарий создаёт только task-local profile/jobs/results, удаляет
точные test-контейнеры и не использует prune.

Только внутри этого синтетического сценария временная PATH-обёртка
направляет bootstrap core tag на настоящий `cpu-test`. Она не меняет
реальные теги, labels, ID или сигналы и исчезает вместе с временным каталогом;
обычный launcher не содержит тестового переключателя.

Здесь не проверяется порядок приватных Bash-функций: сравниваются аргументы
Docker, stdout/stderr, labels, записи attempts и exit code.

Общие исполняемые подмены Docker и Git вынесены в соседний
`tests/support/`, чтобы не дублировать большой shell-текст в Python-файле.
`accept a100-4` проверяется через тот же внешний `./distil3d`: два вызова
дают разные ограниченные отчёты, а missing/broken archive, parser failure,
несовпавшие image ID/label и dirty checkout сохраняют раннюю ошибку и
возвращают ненулевой код. Fake-карта из четырёх A100 проводит всю буквальную
последовательность; отдельные случаи подтверждают foundation fail-fast,
producer→consumer skip, независимую ветку, ошибку selection и остановку
точной активной попытки по Ctrl-C. Профиль с тестовым секретом, `images.tar`,
деревья runs и checkpoints в отчёт не копируются.
После Stage 9.16 точный отчёт содержит 26 случаев и 29 строк. Один native
TE recompute запускается из точного core на первой GPU, без training
record; проверены его ошибка, OOM, сигнал и остановка перед следующим случаем.
Историческая Waymo-часть
различает научный STOP с сохранённым `failed/2`, произвольный код 2, OOM и
непригодную запись; полный операторский depth gate не заменяется новым
comparison. Проверены пропуск зависимых DDW, продолжение независимых
ветвей, сигнал в позднем canary и прекращение кампании при отказе rm/stop.
