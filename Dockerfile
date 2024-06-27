FROM python:3.8

COPY . /app
WORKDIR /app

RUN pip install numpy==1.21.6
RUN pip install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host", "0.0.0.0"]