# docker build -t review_clf .
# docker run -p 8000:8000 review_clf
FROM python:3.12.2-slim

WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r docker_requirements.txt

# Download word vectors
RUN python -c "import gensim.downloader as api; api.load('glove-wiki-gigaword-50')"

# Expose port
EXPOSE 8000

CMD ["python", "src/api.py", "glove-wiki-gigaword-50", "./runs/train1/best_catboost_model.cbm"]
