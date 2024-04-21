# About the application.

## Run the kafka server

Make sure you run your kafka application, zookeeper etc

Open kafka directory and run the following:

```
bin/zookeeper-server-start.sh config/zookeeper.properties
```

Open another terminal and inside the kafka directory

```
bin/kafka-server-start.sh config/server.properties
```

Next create topics and other configurations

```
bin/kafka-topics.sh --create --topic web-logs --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

# Clone the repository

```bash
git clone http://github.com/asifrhaman13/ml_project
```

Create virtual environment.

```bash
virtualenv .venv
```

Next activate the virtual environment.

```bash
source .venv/bin/activate
```

Install the dependencies.

```bash
pip instal -r requirements.txt
```

Run the script.

```bash
uvicorn main:app --reload
```

```bash
chmod +x start.sh
bash start.sh
```

Now you need to start the server which will consume data from kafka.

```bash
uvicorn src.kafka_main:app --reload
```

Run the script to start the kafka server. That will consume all the data.

```bash
python3 kafka_sdk.py
```

In another terminal run the following command:

```bash
uvicorn src.main:app --reload --port=5000
```



### Enter the sample data to train
```http
http://127.0.0.1:8000/items/
```

Body:

```json
{
  "Demand": 15520,
  "Cost": 38,
  "Price": 298.92049116490034,
  "Recession": true,
  "Economy": "Weak",
  "Competition": "High",
  "Market_Size": 1498
}
```


### Predict the output
URL:

```http
http://127.0.0.1:5000/predict/
```

```json
{
  "Demand": [10, 150, 200],
  "Cost": [50, 75, 100],
  "Recession": ["low", "medium", "high"],
  "Economy": ["stable", "unstable", "recession"],
  "Competition": ["low", "medium", "high"],
  "Market_Size": [500, 1000, 1500]
}
```