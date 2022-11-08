FROM python:3.9.15-slim-buster as base
WORKDIR /home
COPY ./ ./
RUN \
# apt-get update && \
# apt-get upgrade -y && \
# apt-get autoremove -y && \
# apt-get clean -y && \
pip install --upgrade pip && \
# pip install wheel && \
pip install -r requirements.txt

FROM base
COPY --from=base ./home/ ./
EXPOSE 8080 8501
CMD ["bash", "./run.sh"]
# CMD ["streamlit", "run", "src/streamlit.py", "&&", "python", "src/api.py"]
# CMD ["tail", "-f", "/dev/null"]