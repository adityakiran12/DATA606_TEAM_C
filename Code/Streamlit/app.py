import pymongo
import streamlit as st
import altair as alt
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from gensim import models, corpora
from wordcloud import WordCloud
import string
import nltk
import re

# Set page layout to wide
st.set_page_config(
     page_title="Sentiment Analysis and Trend Detection in Social Media",
     layout="wide",
     initial_sidebar_state="auto",
)
# Define the file paths for the stock data
file_paths = {
    "Apple": r"df_apple_final_year.csv",
    "Tesla": r"df_nvidea_final_year.csv",
}

text_paths = {
    "Apple": r"sentiments_apple.csv",
    "Tesla": r"sentiments_nvidia.csv",
}



# Function to get current stock price for a given symbol
def get_current_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period='1d')['Close'][0]
        return current_price
    except:
        return None

stock_info = {
    "Apple": {
        "Name": "Apple Inc.",
        "Description": "Apple Inc. designs, manufactures, and markets consumer electronics, software, and services. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
        "Founded": "April 1, 1976, Los Altos, CA",
        "Headquarters": "Cupertino, CA",
        "Founders": ["Steve Jobs", "Steve Wozniak", "Ronald Wayne"],
        "CEO": "Tim Cook (Aug 24, 2011–)",
        "Subsidiaries": ["Apple Store", "Beats Electronics", "Apple Studios", "Beddit", "MORE"],
        "Symbol": "AAPL"
    },
    "Tesla": {
        "Name": "Tesla, Inc.",
        "Description": "Tesla, Inc. specializes in electric vehicles and renewable energy products. It was founded by Elon Musk, Martin Eberhard, JB Straubel, Marc Tarpenning, and Ian Wright in 2003.",
        "Founded": "July 1, 2003, San Carlos, CA",
        "Headquarters": "Austin, TX",
        "Founders": ["Elon Musk", "Martin Eberhard", "JB Straubel", "Marc Tarpenning", "Ian Wright"],
        "CEO": "Elon Musk",
        "Symbol": "TSLA"
    }
    # Add more stock information here for both finance and healthcare sectors
}

client = pymongo.MongoClient("mongodb://localhost:27017/")

# Database names containing the posts
database_names = ["healthcare_data_stream"]

# List to store documents
all_documents = []

for db_name in database_names:
    db = client[db_name]
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        cursor = collection.find({})
        for document in cursor:
            document["subreddit"] = db_name
            all_documents.append({
                "title": document.get("title"),
                "created_utc": pd.to_datetime(document.get("created_utc"), unit='s'),
                "selftext": document.get("selftext")
            })

# Converting list of dicts into a DataFrame
healthcare_df = pd.DataFrame(all_documents)
healthcare_df['text'] = healthcare_df['title'] + ' ' + healthcare_df['selftext']

def clean_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'\(+\)', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ').strip())
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

#data cleaning 
healthcare_df['text_cleaned'] = healthcare_df['text'].apply(clean_text)
#text pre procesing
healthcare_df['text_processed'] = healthcare_df['text_cleaned'].apply(preprocess_text)

# Load the dictionary and LDA model
dictionary = corpora.Dictionary.load("dictionary.gensim")
lda_model = models.LdaModel.load("lda_model.gensim")

# Function to preprocess a single post
def preprocess_post(post):
    tokenized_post = word_tokenize(post.lower())
    tokenized_post = [word for word in tokenized_post if word not in stopwords.words('english')]
    return tokenized_post

# Apply LDA model to each post using apply function
def infer_topic(post):
    tokenized_post = preprocess_post(post)
    bow_post = dictionary.doc2bow(tokenized_post)
    topic_distribution = lda_model.get_document_topics(bow_post)
    dominant_topic = max(topic_distribution, key=lambda x: x[1])
    return dominant_topic[0], dominant_topic[1]

# Apply the function to each row in the dataframe
healthcare_df[['topic', 'topic_probability']] = healthcare_df['text_processed'].apply(infer_topic).apply(pd.Series)

nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to classify sentiment as positive, negative, or neutral
def get_sentiment_label(text):
    sentiment_score = analyzer.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    

healthcare_df['sentiment'] = healthcare_df['text_cleaned'].apply(get_sentiment_label)
finaldf=healthcare_df[['text_processed','topic','sentiment']]

encoder = LabelEncoder()
finaldf['sentiment'] = encoder.fit_transform(finaldf['sentiment'])

vectorizer = joblib.load('vectorizer.pkl')

X_text = vectorizer.transform(finaldf['text_processed'])

finaldf['topic']=finaldf['topic'].astype(str)
finaldf['sentiment']=finaldf['sentiment'].astype(str)

X_numerical = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame(finaldf, columns=['sentiment','topic'])],
                        axis=1)

X_numerical.columns = X_numerical.columns.astype(str)

best_model = joblib.load('final_model_split_0.8.pkl')  # load the model

predictions = best_model.predict(X_numerical)

finaldf['prediction'] = predictions

# Define mappings
label_mapping = {0: "anxiety", 1: "depression"}
topic_mapping = {'0.0': "Seeking Support", '1.0': "Life Events and Relationships", '2.0': "Social Anxiety and Work Challenges", '3.0': "Difficulty with Relationships and Life in General"}
sentiment_mapping = {'0': "Negative", '1': "Neutral", '2': "Positive"}

# Map labels, topics, and sentiments to their corresponding names
finaldf["prediction"] = finaldf["prediction"].map(label_mapping)
finaldf["topic"] = finaldf["topic"].map(topic_mapping)
finaldf["sentiment"] = finaldf["sentiment"].map(sentiment_mapping)

# Function to preprocess text data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load the pre-trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
best_model = joblib.load('final_model_split_0.8.pkl')
lda_model = models.LdaModel.load("lda_model.gensim")
dictionary = corpora.Dictionary.load("dictionary.gensim")

# Load sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define mappings
label_mapping = {0: "anxiety", 1: "depression"}
topic_mapping = {'0': "Seeking Support", '1': "Life Events and Relationships", '2': "Social Anxiety and Work Challenges", '3': "Difficulty with Relationships and Life in General"}
sentiment_mapping = {'0': "Negative", '1': "Neutral", '2': "Positive"}

# Function to classify sentiment as positive, negative, or neutral
def get_sentiment_label(text):
    sentiment_score = analyzer.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# Text input from user


# Sidebar for stock selection
# st.sidebar.header("Select Sector")
selected_sector = st.sidebar.selectbox("Select a sector:", ["Finance", "Healthcare"], index=None)

if selected_sector is None:
    st.write("# Sentiment Analysis and Trend Detection in Social Media")

    # Contributors at the bottom
    st.write("### Contributors:")
    st.write("- Sai Srikar Bollapragada")
    st.write("- Jashwanth Goraka")
    st.write("- Akhilteja Jampani")
    st.write("- Aditya Kiran M")

    st.write("### Guidance:")
    st.write("- Dr. Unal Sakoglu")

if selected_sector == "Finance":

    selected_stock = st.sidebar.selectbox("Select a stock:", ["Apple", "Tesla"])
    
    # Create tabs for data, visualization, and information
    tab2, tab3, tab4 = st.tabs(["Data", "Dashboard", "Text Analysis"])

    # Load stock data based on selection
    file_path = file_paths[selected_stock]
    data = pd.read_csv(file_path)

    # Load text data based on selection
    text_path = text_paths[selected_stock]
    text_data = pd.read_csv(text_path)
    text_data = text_data.dropna(subset=['text'])

    # Function to get current stock price for a given symbol
    def get_current_stock_price(symbol):
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'][0]
            return current_price
        except:
            return None

    selected_stock_info = stock_info[selected_stock]
    symbol = selected_stock_info.get("Symbol")
    if symbol:
        current_price = get_current_stock_price(symbol)
        if current_price is not None:
            selected_stock_info["Stock Price"] = f"${current_price:.2f}"
        else:
            selected_stock_info["Stock Price"] = "Data not available"

        with st.sidebar:
            st.subheader(selected_stock_info["Name"])
            st.write(f"**Description:** {selected_stock_info['Description']}")
            st.write(f"**Founded:** {selected_stock_info['Founded']}")
            st.write(f"**Headquarters:** {selected_stock_info['Headquarters']}")
            if "Founders" in selected_stock_info:
                st.write(f"**Founders:** {', '.join(selected_stock_info['Founders'])}")
            if "CEO" in selected_stock_info:
                st.write(f"**CEO:** {selected_stock_info['CEO']}")
            st.write(f"**Current Stock Price:** {selected_stock_info['Stock Price']}")


        with tab2:
            st.write(f"Displaying data for {selected_stock}:")
            st.dataframe(data)

        with tab3:
            col1, col2 = st.columns(2)

            # Column 1: Graph 1
            with col1:
                st.header("Graph 1")
                stock1 = st.selectbox("Select a stock:", list(file_paths.keys()), key="stock1")
                file_path1 = file_paths[stock1]
                data1 = pd.read_csv(file_path1)

                x_axis1 = st.selectbox("Select X-axis:", list(data1.columns), key="x_axis1")
                y_axis1 = st.selectbox("Select Y-axis:", list(data1.columns), key="y_axis1")
                plot_type1 = st.selectbox("Select plot type:", ["Line", "Scatter", "Bar"], key="plot_type1")

                if plot_type1 == "Line":
                    fig1 = px.line(data1, x=x_axis1, y=y_axis1, title=f"{stock1} - {plot_type1} Plot", width=500, height=400)
                elif plot_type1 == "Scatter":
                    fig1 = px.scatter(data1, x=x_axis1, y=y_axis1, title=f"{stock1} - {plot_type1} Plot", width=500, height=400)
                elif plot_type1 == "Bar":
                    fig1 = px.bar(data1, x=x_axis1, y=y_axis1, title=f"{stock1} - {plot_type1} Plot", width=500, height=400)

                st.plotly_chart(fig1)

            # Column 2: Graph 2
            with col2:
                st.header("Graph 2")
                stock2 = st.selectbox("Select a stock:", list(file_paths.keys()), key="stock2")
                file_path2 = file_paths[stock2]
                data2 = pd.read_csv(file_path2)

                x_axis2 = st.selectbox("Select X-axis:", list(data2.columns), key="x_axis2")
                y_axis2 = st.selectbox("Select Y-axis:", list(data2.columns), key="y_axis2")
                plot_type2 = st.selectbox("Select plot type:", ["Line", "Scatter", "Bar"], key="plot_type2")

                if plot_type2 == "Line":
                    fig2 = px.line(data2, x=x_axis2, y=y_axis2, title=f"{stock2} - {plot_type2} Plot", width=500, height=400)
                elif plot_type2 == "Scatter":
                    fig2 = px.scatter(data2, x=x_axis2, y=y_axis2, title=f"{stock2} - {plot_type2} Plot", width=500, height=400)
                elif plot_type2 == "Bar":
                    fig2 = px.bar(data2, x=x_axis2, y=y_axis2, title=f"{stock2} - {plot_type2} Plot", width=500, height=400)

                st.plotly_chart(fig2)

        # Text Analysis tab
        with tab4:
            # Display word cloud
            st.header(f"Word Cloud for {selected_stock}")
            text = ' '.join(text_data['text'].values)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text[:4000])

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

            # Bar Chart of Sentiment Labels
            st.header("Bar Chart of Sentiment Labels")
            sentiment_counts = text_data['label'].value_counts()
            plt.figure(figsize=(8, 6))
            sentiment_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.title('Distribution of Sentiment Labels')
            plt.xlabel('Sentiment Label')
            plt.ylabel('Frequency')
            plt.xticks(rotation=0)
            st.pyplot(plt)




elif selected_sector == "Healthcare":

    htab1, htab2, htab3 = st.tabs(["Data", "Dashboard", "Text Analysis"])

    with htab1:
        st.write("Displaying data for Healthcare sector:")
        st.dataframe(finaldf)

    with htab2:

        # Divide the layout into three columns
        columns = st.columns(4)

        with columns[0]:
            st.write("total number of posts: ")
            st.write(f"# {finaldf.shape[0]}") 
        # Display prediction distribution in the first column
        with columns[1]:
            st.write("### Prediction Distribution")
            prediction_counts = finaldf['prediction'].value_counts()
            st.dataframe(prediction_counts)

        # Display topic distribution in the second column
        with columns[2]:
            st.write("### Sentiment Distribution")
            sentiment_counts = finaldf['sentiment'].value_counts()
            st.dataframe(sentiment_counts)


        # Display sentiment distribution in the third column
        with columns[3]:
            st.write("### Topic Distribution")
            topic_counts = finaldf['topic'].value_counts()
            st.dataframe(topic_counts)


        text = ' '.join(finaldf['text_processed'].values)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    with htab3:
        user_text = st.text_area("Enter your text here:")

        # Button to perform analysis
        if st.button("Analyze"):
            # Data preprocessing
            cleaned_text = clean_text(user_text)
            preprocessed_text = preprocess_text(cleaned_text)
            
            # Sentiment analysis
            sentiment = get_sentiment_label(preprocessed_text)
            encoder = LabelEncoder()
            sentiment = encoder.fit_transform([sentiment])[0]
            
            # Apply LDA model to each post using apply function
            def infer_topic(post):
                tokenized_post = preprocess_text(post)
                bow_post = dictionary.doc2bow(tokenized_post.split())
                topic_distribution = lda_model.get_document_topics(bow_post)
                dominant_topic = max(topic_distribution, key=lambda x: x[1])
                return dominant_topic[0], dominant_topic[1]

            # Apply the function to the user text
            topic, topic_probability = infer_topic(preprocessed_text)

            # Prediction using pre-trained model
            X_text = vectorizer.transform([preprocessed_text])
            X_text_df = pd.DataFrame(X_text.toarray())
            X_numerical = pd.concat([X_text_df, pd.DataFrame({'sentiment': sentiment, 'topic': str(topic)}, index=[0])], axis=1)
            X_numerical.columns = X_numerical.columns.astype(str)

            # Predict label
            prediction = best_model.predict(X_numerical)
            label = label_mapping[prediction[0]]

            # Display result
            st.header("Analysis Result:")
            st.markdown(f"**Emotional State:** {label}")
            st.markdown(f"**Sentiment:** {sentiment_mapping[str(sentiment)]}")
            st.markdown(f"**Topic:** {topic_mapping[str(topic)]} (Probability: {topic_probability:.2f})")
