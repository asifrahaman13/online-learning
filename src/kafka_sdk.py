import csv
# Import the required packages and libraries.
from kafka import KafkaProducer
from confluent_kafka import Consumer, KafkaError
import time
from src.model_creation import train_model


# Logger class to log the request and response.
class Logger:
    def log_request_and_response(self, request_url, request_method, data):
        print(f"Request: {request_method} {request_url}")
        print(f"Request Headers: {data}")


# KafkaHandler class to send the request to Kafka.
class KafkaHandler:
    """Initialize the KafkaHandler class with the Kafka bootstrap servers."""

    def __init__(self, kafka_bootstrap_servers):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.producer = KafkaProducer(bootstrap_servers=self.kafka_bootstrap_servers)

    def send_to_kafka(self, key, value):
        import json

        value = json.dumps(value)
        self.producer.send("web-logs", key=key.encode(), value=value.encode())


class CsvFile:

    def __init__(self, file_path):
        self.file_path = file_path

    def append_to_csv(
        self, demand, cost, price, recession, economy, competition, market_size
    ):
        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [demand, cost, price, str(recession), economy, competition, market_size]
            )


# TrafficProcessingSDK class to process the request and consume from Kafka.
class TrafficProcessingSDK:
    """Initialize the TrafficProcessingSDK class with the Kafka bootstrap servers and group id."""

    def __init__(self, kafka_bootstrap_servers, group_id):
        self.logger = Logger()
        self.kafka_handler = KafkaHandler(kafka_bootstrap_servers)
        self.group_id = group_id

    """Pocess the request to send to kafka handler."""

    def process_request(self, url, req_type, data):
        print(url, req_type, data)
        self.logger.log_request_and_response(url, req_type, data)

        self.kafka_handler.send_to_kafka(
            "request",
            data,
        )

    def consume_from_kafka(self):

        # Create a consumer class instance.
        consumer = Consumer(
            {
                "bootstrap.servers": self.kafka_handler.kafka_bootstrap_servers,
                "group.id": self.group_id,
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe(["web-logs"])

        def msg_callback(msg):
            import json

            print(msg.value().decode("utf-8"))
            msg_str = msg.value().decode("utf-8")
            msg_dict = json.loads(msg_str)
            print(type(msg_dict))
            print(msg_dict)
            print(msg_dict["Demand"])

            csv_operation = CsvFile("service_pricing_data.csv")

            demand = msg_dict["Demand"]
            cost = msg_dict["Cost"]
            price = msg_dict["Price"]
            recession = msg_dict["Recession"]
            economy = msg_dict["Economy"]
            competition = msg_dict["Competition"]
            market_size = msg_dict["Market_Size"]

            csv_operation.append_to_csv(
                demand, cost, price, recession, economy, competition, market_size
            )
            print("Received message: {}".format(msg.value().decode("utf-8")))

            train_model()

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(msg.error())
                        break

                msg_callback(msg)
        finally:
            consumer.close()


# Example usage
if __name__ == "__main__":
    kafka_bootstrap_servers = "localhost:9092"
    group_id = "traffic-processing-group"
    sdk = TrafficProcessingSDK(kafka_bootstrap_servers, group_id)

    while True:
        url = input("Enter the server URL: ")
        url = "http://localhost:8000"
        data = {
            "Demand": 152,
            "Cost": 38,
            "Price": 298.92049116490034,
            "Recession": True,
            "Economy": "Weak",
            "Competition": "High",
            "Market_Size": 1498,
        }

        sdk.process_request(url, "GET", data)
        time.sleep(1)
        sdk.consume_from_kafka()
