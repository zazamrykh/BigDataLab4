# docker build -t review_clf .
# docker run -p 8000:8000 review_clf
FROM python:3.12.2-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r docker_requirements.txt
RUN python -c "import gensim.downloader as api; api.load('glove-wiki-gigaword-50')"

CMD ["python", "src/api.py", "glove-wiki-gigaword-50", "./models/best_catboost_model.cbm"]
