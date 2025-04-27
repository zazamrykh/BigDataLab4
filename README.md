Lab 2. Integration with postgreeSql.

How to check if working?
1) Run command:
docker-compose down && docker-compose build --no-cache && docker-compose up -d
2) Health check:
curl http://localhost:8000/health
3) Post query:
curl -v -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"summary": "Great product!", "text": "This product works perfectly and I love it.", "HelpfulnessNumerator": 5, "HelpfulnessDenominator": 7}'
4) Check if prediction is added to database:
curl http://localhost:8000/predictions