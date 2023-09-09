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

## Data visualization

![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/Data_Visualization.png)
![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/heatmap.png)

## Performance evaluation

![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/LogisticRegression.png)
![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/svc.png)
![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/DecisionTreeClassifier.png)
![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/CatBoostClassifier.png)
![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/RandomForestClassifier.png)
![Alt text](https://github.com/Abbey-12/ML-Anomaly-Detection/blob/main/Image/LogisticRegression.png)

## Comment

Different models' performance is assessed using various metrics. The selection of metrics in model comparison depends on the specific goals and implications linked to different error types. Precision is the metric of choice when minimizing false alarms is paramount, while recall takes precedence when capturing all genuine anomalies is crucial. The F1 score provides a balanced assessment encompassing both precision and recall. Accuracy is appropriate for balanced datasets with minimal error consequences but may lack informativeness in imbalanced data scenarios. The most important metric depends on our application's requirements, and the costs of false positives and false negatives. In this scenario with a balanced dataset, accuracy serves as the comparison metric, with the logistic regressor outperforming other models, achieving an 85.8% accuracy rate. 
