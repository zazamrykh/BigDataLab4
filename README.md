# Лабораторная работа 2: Интеграция с PostgreSQL

## Локальное тестирование

Как проверить работу приложения:

1) Запустить контейнеры:
```bash
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

2) Проверить доступность API:
```bash
curl http://localhost:8000/health
```

3) Отправить запрос на предсказание:
```bash
curl -v -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"summary": "Great product!", "text": "This product works perfectly and I love it.", "HelpfulnessNumerator": 5, "HelpfulnessDenominator": 7}'
```

4) Проверить, что предсказание добавлено в базу данных:
```bash
curl http://localhost:8000/predictions
```

## CI/CD Pipeline

В этом проекте реализован CI/CD pipeline с использованием GitHub Actions и Docker.

### Структура проекта

- `src/` - исходный код приложения
- `tests/` - тесты
- `.github/workflows/` - конфигурация CI/CD pipeline
- `docker-compose.yml` - конфигурация для запуска контейнеров
- `Dockerfile` - инструкции для сборки Docker-образа

### CI Pipeline (Continuous Integration)

CI pipeline автоматически запускается при:
- Пуше в ветку main
- Создании Pull Request в ветку main
- Ручном запуске через GitHub Actions
- По расписанию (каждый понедельник в 00:00)

#### Этапы CI Pipeline:

1. **Checkout репозитория** - получение кода из репозитория
2. **Установка зависимостей** - установка необходимых Python-пакетов
3. **Запуск модульных тестов** - проверка корректности работы отдельных компонентов
4. **Сборка Docker-образа** - создание образа с приложением
5. **Публикация образа в DockerHub** - отправка образа в публичный реестр

### CD Pipeline (Continuous Deployment)

CD pipeline автоматически запускается:
- По расписанию (каждый понедельник в 8:00 утра)
- При ручном запуске через GitHub Actions

#### Этапы CD Pipeline:

1. **Checkout репозитория** - получение кода из репозитория
2. **Создание конфигурации** - подготовка файла .env
3. **Загрузка Docker-образа** - получение образа из DockerHub
4. **Запуск контейнеров** - запуск приложения и базы данных с помощью docker-compose
5. **Функциональное тестирование** - проверка работоспособности API и базы данных
6. **Остановка контейнеров** - корректное завершение работы

### Функциональное тестирование

В рамках CD pipeline выполняются следующие функциональные тесты:

1. **Проверка доступности API** - запрос к эндпоинту /health
2. **Проверка работы модели** - отправка данных на эндпоинт /predict
3. **Проверка сохранения в базе данных** - запрос к эндпоинту /predictions

### Инструкции по использованию

#### Настройка секретов в GitHub

Для работы CI/CD pipeline необходимо настроить следующие секреты в репозитории GitHub:

- `DOCKERHUB_USERNAME` - имя пользователя в DockerHub
- `DOCKERHUB_TOKEN` - токен доступа к DockerHub

#### Ручной запуск CI Pipeline

1. Перейдите в раздел "Actions" в репозитории GitHub
2. Выберите workflow "Build, Test and Push Docker Image"
3. Нажмите "Run workflow"
4. Выберите ветку и нажмите "Run workflow"

#### Ручной запуск CD Pipeline

1. Перейдите в раздел "Actions" в репозитории GitHub
2. Выберите workflow "Deploy and Test Model Container"
3. Нажмите "Run workflow"
4. Нажмите "Run workflow"

### Приложение к отчету

В отчет по лабораторной работе необходимо включить:

1. Скриншоты успешного выполнения CI/CD pipeline
2. Конфигурационные файлы CI/CD pipeline (.github/workflows/ci.yml и .github/workflows/cd.yml)
3. Описание внесенных изменений в проект для поддержки CI/CD