FROM on611/recommend-shopfee:1.0.1
WORKDIR /recommend
COPY . /recommend

SHELL ["conda", "run", "-n", "pytorch-cpu", "/bin/bash", "-c"] 
EXPOSE 5000
ENTRYPOINT ["conda", "run", "-n", "pytorch-cpu", "python", "main.py"] 
