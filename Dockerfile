FROM jupyter/scipy-notebook:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=1111", "--no-browser", "--allow-root", "--notebook-dir=/app", "--NotebookApp.token=''", "--NotebookApp.password=''"]