a) Preprocessing

We initiated our data collection process by utilizing the PRAW (Python Reddit API Wrapper) library to extract data from relevant healthcare-related subreddits, including r/anxiety, r/depression, r/mentalhealth, and r/socialanxiety. The extracted data was stored in a MongoDB database for subsequent analysis.

Upon retrieving the data from MongoDB, we performed initial data cleaning steps to ensure data integrity. We first examined the 'title' and 'selftext' columns for null values, removing rows where both these columns were null. Additionally, we identified and removed rows where the author was either 'removed' or 'deleted', indicative of irrelevant or inappropriate content. Furthermore, we filtered out rows containing '[removed]' or '[deleted]' within the 'selftext' column, as well as rows with a score of 1 where the 'selftext' was removed or deleted.

To facilitate analysis, we extracted temporal features from the 'created_utc' column, including month, day, and year. Subsequently, we concatenated the 'title' and 'selftext' columns into a single 'fulltext' column to consolidate textual information. Following this, we retained essential columns for analysis, including 'created_utc', 'id', 'name', 'score', 'fulltext', 'author', 'subreddit', 'month', 'day', and 'year'.

Further cleaning of the 'fulltext' column was performed using a custom function to remove noise and irrelevant information. This involved converting text to lowercase, removing text within brackets, emojis, additional parentheses, punctuation, newline characters, URLs, hashtags, and mentions.

Subsequently, the preprocessed text was tokenized, converted to lowercase, and lemmatized to ensure consistency in word forms and reduce dimensionality. Stop words and non-alphabetic tokens were removed from the tokenized text. The resulting preprocessed text served as input for subsequent analyses.

b) Data Analysis

Following data preprocessing, we conducted exploratory data analysis (EDA) to gain deeper insights into the nature of the discussions within healthcare-related subreddits. This analysis aimed to uncover dominant themes, language trends, and post characteristics, laying the foundation for subsequent analysis and interpretation.

As part of our exploratory data analysis (EDA), we employed various text analysis techniques to gain insights into the discussions within healthcare-related subreddits. Initially, we generated word clouds to visualize the most frequently occurring words, facilitating the identification of dominant themes among subreddit users.

Additionally, n-gram analysis was conducted to uncover subtle language trends and sequential pattern

Furthermore, a histogram of word counts in each post was visualized to provide an overview of word distribution across different posts in the dataset. This analysis guided the implementation of tailored text processing techniques to accommodate varying post lengths for accurate analysis and interpretation.

Subsequently, we performed topic modeling using Latent Dirichlet Allocation (LDA) to extract latent topics from the text data. LDA revealed four optimal topics within our dataset, each characterized by distinct themes and language patterns:

Topic 0: Mental Health

Topic 1: Difficulty and Uncertainty

Topic 2: Social Connections and Time

Topic 3: Anxiety and Daily Life

The LDA analysis proved to be a pivotal discovery in our healthcare project, providing insightful points and guiding subsequent analyses and classifications.

Finally, sentiment analysis was conducted on the preprocessed text data using the VADER (Valence Aware Dictionary and Sentiment Reasoner) lexicon. This analysis enabled the identification of positive, negative, and neutral posts, facilitating a deeper understanding of sentiment dynamics within the discussions.

c) Classification

Following manual annotation, keyword extraction, and other labeling techniques, we classified all 120k posts as either indicative of depression or anxiety. To ensure labeling accuracy, we distributed the sampled posts among ourselves for validation.

Subsequently, we employed various classification models, including RandomForest, SVM, DecisionTree, and the BERT transformer model, to classify the labeled data. Exploring different train-test-validation splits, including [0.75, 0.8, 0.85], we determined that a split of 0.75 yielded optimal results. Notably, the BERT model outperformed others, achieving an impressive F1 score of 0.91, attributed to the Next Sentence Prediction (NSP) mechanism inherent in BERT.

d) Deployment

With the model trained and validated, we implemented real-time data streaming from Reddit using Kafka, fetching data every six hours. Subsequently, Streamlit was utilized to automate essential Extract, Transform, Load (ETL) techniques, feeding processed data into the model. The resulting Streamlit-based user interface (UI) provides interactive visualizations and text analysis for real-time data, allowing users to filter data by time intervals such as the last 24 hours or 6 hours.
