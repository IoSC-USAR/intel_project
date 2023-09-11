FROM docker.io/hdgigante/python-opencv

RUN dnf install python python-pip -y
WORKDIR /usr/src/app

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "api_testing.py"]
