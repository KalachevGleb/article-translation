# Article Translation System Design

## Цель
Автоматизированный переводчик научных статей в формате LaTeX с использованием LLM (GPT-o3/o3-mini от OpenAI).

## Базовая пара языков
Русский → Английский (с возможностью расширения)

## Архитектура системы

### Модули

1. **LaTeX Parser** (`latex_parser.py`)
   - Flattening TeX документа (разворачивание `\input`, `\include`)
   - Выделение структуры: секции, подсекции, абзацы
   - Извлечение формул (inline: `$...$`, display: `$$...$$`, `\[...\]`, `equation`, `align` и т.д.)

2. **Dependency Analyzer** (`dependency_analyzer.py`)
   - Анализ связей между секциями через LLM
   - Построение графа зависимостей
   - Определение порядка перевода

3. **Terminology Manager** (`terminology_manager.py`)
   - Извлечение терминов через LLM
   - Векторная база переводов с контекстами (embeddings)
   - API для интерактивного уточнения переводов
   - Автоматический/ручной режимы

4. **Translation Engine** (`translation_engine.py`)
   - Перевод по секциям с учетом зависимостей
   - Контекст: зависимые секции + словарь терминов
   - Строгое правило: формулы не изменяются
   - Естественный перевод текста с перестройкой фраз

5. **Formula Validator** (`formula_validator.py`)
   - Извлечение формул из исходного и переведенного текста
   - Сравнение на уровне абзацев:
     - Inline формулы: совпадение с точностью до перестановки
     - Display формулы: сохранение порядка
   - Повторный перевод при расхождениях (макс. 2 попытки)
   - Маркировка проблемных абзацев: `{\color{red}...\footnote{diff}}`

6. **Report Generator** (`report_generator.py`)
   - HTML отчет о процессе перевода
   - Статистика: количество секций, терминов, исправлений
   - Список проблемных мест
   - Exit code: 0 (успех) или код ошибки

7. **OpenAI Client** (`openai_client.py`)
   - Обертка над OpenAI API
   - Управление запросами к GPT-o3/o3-mini
   - Обработка ошибок и ретраи

## Алгоритм работы

### Фаза 1: Подготовка
```
1. Flatten LaTeX документа
2. Парсинг структуры → список секций S = {s₁, s₂, ..., sₙ}
3. Извлечение формул для каждой секции
```

### Фаза 2: Анализ зависимостей
```
4. LLM запрос: определение зависимостей между секциями
   Input: список названий и кратких содержаний секций
   Output: граф зависимостей G = (S, E)
5. Топологическая сортировка для порядка перевода
```

### Фаза 3: Терминология
```
6. LLM запрос: извлечение терминов из всего документа
   Output: список пар (термин_ru, предложенный_перевод_en)
7. Поиск похожих терминов в векторной БД (по embedding контекста)
8. [Опционально] Взаимодействие с пользователем для уточнения
9. Финальный словарь терминов D = {term_ru: term_en}
```

### Фаза 4: Перевод
```
10. Для каждой секции sᵢ в порядке топологической сортировки:
    a. Получить переводы зависимых секций: T(dependencies(sᵢ))
    b. LLM запрос перевода:
       - Context: T(dependencies(sᵢ)) + словарь D
       - Prompt: переводить текст естественно, НЕ ИЗМЕНЯТЬ формулы
       - Output: переведенная секция tᵢ
    c. Сохранить tᵢ
```

### Фаза 5: Валидация формул
```
11. Для каждого абзаца p в исходной и переведенной версиях:
    a. Извлечь формулы: F_src(p) и F_trans(p)
    b. Проверить:
       - Inline: set(F_inline_src) == set(F_inline_trans) (с точностью до перестановки)
       - Display: list(F_display_src) == list(F_display_trans) (порядок важен)
    c. Если не совпадает:
       - Попытка 1: повторный перевод с усиленным промптом
       - Попытка 2: еще раз с максимальной строгостью
       - Если неудача: маркировка {\color{red}...\footnote{diff}}
```

### Фаза 6: Генерация отчета
```
12. Сборка переведенного документа
13. Генерация HTML отчета:
    - Статистика перевода
    - Список терминов
    - Проблемные места
    - Время выполнения
14. Return exit code
```

## Структура проекта

```
article-translation/
├── article_translator/
│   ├── __init__.py
│   ├── latex_parser.py
│   ├── dependency_analyzer.py
│   ├── terminology_manager.py
│   ├── translation_engine.py
│   ├── formula_validator.py
│   ├── report_generator.py
│   ├── openai_client.py
│   └── models.py  # Dataclasses для структур данных
├── tests/
│   └── ...
├── examples/
│   └── sample_article.tex
├── config.yaml  # Конфигурация (API ключи, модели, параметры)
├── requirements.txt
├── setup.py
└── README.md
```

## Конфигурация

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: "o3-mini"  # или "o3"
  temperature: 0.3

translation:
  source_language: "russian"
  target_language: "english"
  max_retries: 2
  auto_mode: true  # false для интерактивного уточнения терминов

terminology:
  database_path: "terms.db"
  embedding_model: "text-embedding-3-large"
  similarity_threshold: 0.85

output:
  report_format: "html"
  mark_problematic: true
  color: "red"
```

## API интерфейс

```python
from article_translator import ArticleTranslator

translator = ArticleTranslator(config_path="config.yaml")

result = translator.translate(
    source_file="article.tex",
    output_file="article_en.tex",
    terminology_mode="auto"  # или "interactive"
)

print(f"Exit code: {result.exit_code}")
print(f"Report: {result.report_path}")
```

## Промпты LLM

### Dependency Analysis
```
Проанализируй структуру научной статьи. Определи логические зависимости между секциями:
какие секции используют понятия/результаты из других секций.
Верни JSON с графом зависимостей.
```

### Terminology Extraction
```
Извлеки все специфичные термины из научной статьи и предложи их перевод на английский.
Фокус на математических, научных и доменных терминах.
```

### Translation
```
Переведи секцию с русского на английский. Требования:
1. НЕ ИЗМЕНЯЙ формулы (в $...$ и $$...$$)
2. Переводи текст естественно, перестраивай фразы для естественности
3. Используй предоставленный словарь терминов
4. Учитывай контекст из предыдущих секций
```

### Formula-strict Translation (retry)
```
Повтори перевод. КРИТИЧНО: формулы должны остаться ИДЕНТИЧНЫМИ исходным.
Проверь каждую формулу дважды перед выводом.
```
