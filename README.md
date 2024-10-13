# Text Classifier Application

Это приложение предназначено для классификации текстовых документов в различные категории (кластеры) с использованием моделей машинного обучения и библиотек NLP.

## Содержание

- [Требования](#требования)
- [Установка](#установка)
- [Запуск приложения](#запуск-приложения)
- [Использование](#использование)


## Требования

- Python 3.7 или выше
- Uvicorn
- FastAPI
- Scikit-learn
- NLTK
- Sentence Transformers

## Установка

1. Клонируйте репозиторий на свой компьютер:
    ```bash
    git clone https://github.com/K0BERZ/textClassifier.git
    cd textClassifier
    ```

2. Создайте и активируйте виртуальное окружение:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Для Windows используйте `venv\Scripts\activate`
    ```

3. Установите необходимые библиотеки:
    ```bash
    pip install -r requirements.txt
    ```
   
## Запуск приложения

После установки всех зависимостей и необходимых ресурсов запустите приложение с помощью команды:

```bash
uvicorn app.main:app --reload
```

## Использование

1.	Откройте браузер и перейдите по адресу http://127.0.0.1:8000.
2. На главной странице загрузите текстовый файл, который хотите классифицировать, с помощью кнопки “Выберите файл”.
3.	Нажмите кнопку “Отправить”. После обработки файла, вы получите результат классификации, который отображает номер кластера и его описание.


## Описание классов

* Класс 0 (Финансовые и экономические новости): Описание включает темы, связанные с экономической ситуацией, изменениями в валютных курсах, а также результатами и прогнозами крупных компаний и отраслей. 
* Класс 1 (Технологические новости и инновации): Включает тексты, которые обсуждают технологические достижения, новые продукты и услуги, а также изменения в индустрии технологий.
* Класс 2 (Политические и социальные события): Охватывает актуальные политические события, конфликты и изменения в социальной сфере.
* Класс 3 (Спортивные события и достижения): Фокусируется на спортивных событиях и достижениях отдельных спортсменов и команд.
