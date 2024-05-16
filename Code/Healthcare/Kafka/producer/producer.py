import praw
from confluent_kafka import Producer
import json
import time

def fetch_reddit_posts():
    
    reddit = praw.Reddit(
        client_id="oxPG2v2-C79UklvB80BCAQ",
        client_secret="XGRjRYrgTYJMIaeUgriEJQFzV4rlcQ",
        user_agent="Srikar_1709",
    )

    #Kafka broker
    kafka_config = {
        "bootstrap.servers": "localhost:9092",
    }
    producer = Producer(kafka_config)
    
    subreddits = ["depression", "anxiety", "socialanxiety", "mentalhealth"]

    while True:
        try:
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                # Fetch the last 100 posts
                for post in subreddit.new():
                    post_data = {
                        "subreddit": subreddit_name,
                        "title": post.title,
                        "created_utc": post.created_utc,
                        # "author": post.author.name,
                        "selftext":post.selftext
                    }
                    message = json.dumps(post_data)
                    producer.produce("reddit-posts", value=message)

            producer.flush()
            print("Fetched and published the lastest Reddit posts from multiple subreddits. Sleeping for 6 hours...")
            time.sleep(21600)  # Sleep for 6 hrs
        except Exception as e:
            print(f"Error fetching Reddit posts: {e}")

if __name__ == "__main__":
    fetch_reddit_posts()
