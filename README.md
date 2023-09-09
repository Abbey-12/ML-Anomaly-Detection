# ML-Anomaly-Detection
Network traffic anomaly detection with machine learning


## Building the Docker Image

```bash
docker build -t ml_model .

```
## Running the Docker Container


```bash
docker run -p 80:80 ml_model

```
## IF you want to save the models to local machine

```bash
docker run -p 80:80 -v /path/to/local/models:/app/models ml_model

```
