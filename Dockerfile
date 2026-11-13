FROM python:3.12-slim

WORKDIR /app

COPY fim_algorithm.py train_fim.py fim_nengo.py fim_ui.py ./

ENV PORT=8080 HOST=0.0.0.0 PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python3", "fim_ui.py"]
