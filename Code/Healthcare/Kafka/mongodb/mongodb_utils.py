from pymongo import MongoClient
#the client
client = MongoClient("mongodb://localhost:27017/")

db = client["healthcare_data_stream"]  

all_documents = []
cursor = db.posts.find({})  

for document in cursor:
    title = document.get("title")
    # author = document.get("author")
    selftext= document.get("selftext")
    created_utc= document.get("created_utc")
    post_data = {"title": title,"selftext":selftext,"created_utc":created_utc}
    all_documents.append(post_data)

