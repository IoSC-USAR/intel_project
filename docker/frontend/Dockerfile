FROM docker.io/hdgigante/python-opencv
RUN dnf install python python-pip -y

WORKDIR /usr/src/app

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
