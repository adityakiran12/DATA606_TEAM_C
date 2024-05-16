from confluent_kafka import Consumer, KafkaError
import json
from pymongo import MongoClient
from datetime import datetime

def consume_reddit_posts():
    # MongoDB setup
    mongo_client = MongoClient("mongodb://localhost:27017/") 
    db = mongo_client.healthcare_data_stream  
    
    # Kafka Consumer setup
    kafka_config = {
        'bootstrap.servers': 'localhost',  
        'group.id': 'reddit-posts-consumer-group',  
        'auto.offset.reset': 'earliest'  
    }

    consumer = Consumer(kafka_config)
    consumer.subscribe(['reddit-posts'])  

    try:
        while True:
            msg = consumer.poll(timeout=5.0)  

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break

            # Message is a normal message
            post_data = json.loads(msg.value().decode('utf-8'))
            # Generate a new collection name based on timestamp
            collection_name = "posts_" + datetime.now().strftime("%Y%m%d%H%M%S")
            collection = db[collection_name]  
            #Insert data into MongoDB
            collection.insert_one(post_data)
            print("Inserted post into MongoDB:", post_data['title'])

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == "__main__":
    consume_reddit_posts()
