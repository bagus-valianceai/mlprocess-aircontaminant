FROM python:3.9.15-slim-buster
WORKDIR /home
COPY ./requirements.txt ./
RUN \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "src/api.py"]