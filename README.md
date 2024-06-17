## Create image

docker-compose up -d

## Example api

curl -X POST http://127.0.0.1:8001/ner -H "Content-Type: application/json" -d '{"text_list": ["example text"]}'
