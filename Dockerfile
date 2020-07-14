FROM python:3.6

WORKDIR /usr/src/app

COPY ./ /usr/src/app

RUN python3 -m venv /opt/venv

COPY requirements.txt .

RUN /opt/venv/bin/pip3 install -r requirements.txt

COPY flask_demo_docker/demo.py .

CMD ["/opt/venv/bin/python", "demo.py"]