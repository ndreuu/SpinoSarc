# Отдельное пространство разработки SpinoSarc

Это самостоятельный checkout [ndreuu/SpinoSarc](https://github.com/ndreuu/SpinoSarc).
У него собственная `.git` и рабочая ветка `codex/spinosarc-workspace`, созданная
от проверенного демо `codex/lumbar-mri-demo` (`9395f13`). Он не является
подмодулем исходного проекта. `main` сохраняет upstream-основу.

## Где что хранить

```text
spinosarc_app/             код приложения и вычисления
demo/scripts/             запуск, модели, данные, воспроизводимые проверки
demo/tests/               тесты инструментов
demo/requirements-*.lock   закреплённые зависимости
docs/                     описание текущей архитектуры и разработки
research/
  plans/                  цели, контракты и этапы будущей работы
  tools/                  обзоры инструментов и измерений
  architecture/           исследовательская карта компонентов
  datasets/               источники данных и ограничения
  archive/parent_project/  история прежнего backend
paper/                    исходная публикация SpinoSarc
.spinosarc.local.json      локальная настройка runtime, игнорируется Git
```

Файлы исследований скопированы из исходного проекта; оригиналы сохранены.
Их состояние зафиксировано на 08.10.2026. Реализованный FastAPI/backend,
его ROI и эксперименты описаны как история другого проекта. Действующее
состояние приложения описывают [архитектура](architecture_ru.md) и
[инструкция демо](../demo/README.md), а не исследовательские гипотезы.

Для VS Code открыть `SpinoSarc.code-workspace` в корне. Поиск исключает
окружения, снимки, веса и результаты; файлы остаются доступными на диске.

## Запустить подготовленный пример

В этом рабочем каталоге `.spinosarc.local.json` уже подключает существующий
runtime с окружениями, весами, открытым Sudirman и рассчитанным кешем.
Из корня нового checkout:

```sh
python3 demo/scripts/setup_spinosarc.py --verify-only
python3 demo/scripts/run_spinosarc.py --example
```

Код `spinosarc_app` исполняется из нового checkout; адаптеры TSS по умолчанию
также выбираются из него. Явные `SPINE_TSS_*` переопределения сохраняются. Тяжёлые
зависимости, модели, снимки и результаты физически пока остаются в прежнем
runtime. Значение пути хранится только в игнорируемом локальном JSON.
Этот каталог должен оставаться доступным для такого способа запуска.
Проверка `--verify-only` не устанавливает пакеты и не перезаписывает старое `.app`.

Порядок выбора runtime:

1. Переменная `SPINOSARC_RUNTIME_ROOT`.
2. Поле `runtime_root` файла `.spinosarc.local.json`.
3. Корень текущего репозитория.

Пример формата — [demo/runtime.example.json](../demo/runtime.example.json).
Относительный путь в JSON считается от корня репозитория, а не от текущего
каталога терминала. JSON читается как данные. Не добавлять его локальную
копию в Git. Путь текущего runtime можно увидеть без запуска моделей:

```sh
python3 -c 'import sys; sys.path.insert(0, "demo/scripts"); from _demo_paths import runtime_root; print(runtime_root())'
```

## Полностью отдельный runtime в будущем

Создать собственный runtime можно штатными инструментами без зависимости
от исходного проекта. Задать его в локальном JSON либо переменной окружения,
затем выполнить [подготовку с нуля](../demo/README.md). Для Python 3.12:

```sh
python3.12 demo/scripts/setup_spinosarc.py --setup-tss
python3.12 demo/scripts/fetch_totalspineseg.py --workers 2 --range-workers 2
python3.12 demo/scripts/setup_spinosarc_musclemap.py
python3.12 demo/scripts/fetch_spinosarc_example.py
python3.12 demo/scripts/run_spinosarc.py --example
```

Это устанавливает окружения и скачивает модели/пример в выбранный runtime.
Существующий runtime сейчас подключён для быстрого продолжения разработки.

## Ветки и изменения

`origin` — наш форк, `upstream` — оригинальный `neuromath/SpinoSarc`.
Для следующего изменения создать ветку от текущего рабочего пространства:

```sh
git switch -c codex/next-change
git remote -v
git diff --check
```

Материалы исследования добавлять в `research/`, действующую документацию —
в `docs/`, код — в соответствующий модуль приложения или `demo/scripts/`.
Для нового исследования указать дату, вопрос, источники, проверенный результат
и открытые вопросы. Предполагаемые возможности не помечать реализованными.

Проверки приложения описаны в [demo/README.md](../demo/README.md).
Для правки только Markdown достаточно проверить ссылки и `git diff --check`.
Оригинальная лицензия, публикация, citation и история upstream сохранены.
