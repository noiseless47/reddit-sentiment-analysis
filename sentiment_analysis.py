import pandas as pd
import streamlit as st
import seaborn as sns
from transformers import pipeline
import praw  # For Reddit API
from dotenv import load_dotenv
import os
import nltk
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake  # Import RAKE for keyword extraction
from wordcloud import WordCloud  # For word clouds

# Load environment variables from .env file
load_dotenv()

# Download NLTK resources
nltk.download('punkt')  # Ensure punkt is downloaded
nltk.download('punkt_tab')  # Download punkt_tab to avoid LookupError
nltk.download('stopwords')  # Download stopwords

# Function to collect data from Reddit
def collect_reddit_data(subreddit, limit=100):
    reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                         client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                         user_agent=os.getenv('REDDIT_USER_AGENT'))
    subreddit = reddit.subreddit(subreddit)
    posts = [{'text': post.title, 
               'created_utc': post.created_utc, 
               'author': post.author, 
               'score': post.score, 
               'url': f'{post.permalink}'} for post in subreddit.new(limit=limit)]
    return pd.DataFrame(posts)

# Function to extract aspects using RAKE
def extract_aspects(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()  # Returns a list of keywords

# Aspect-Based Sentiment Analysis
def aspect_based_sentiment(review, aspect_keywords, sentiment_pipeline):
    sentences = sent_tokenize(review)
    aspect_sentiments = {aspect: [] for aspect in aspect_keywords}  # Initialize a list for each aspect
    
    for sentence in sentences:
        for aspect in aspect_keywords:
            if aspect in sentence.lower():  # Case insensitive
                sentiment = sentiment_pipeline(sentence)[0]['label']
                aspect_sentiments[aspect].append(sentiment)
    
    # Convert lists of sentiments to a single sentiment for each aspect
    for aspect in aspect_sentiments:
        if aspect_sentiments[aspect]:  # If there are sentiments for this aspect
            aspect_sentiments[aspect] = max(set(aspect_sentiments[aspect]), key=aspect_sentiments[aspect].count)
        else:
            aspect_sentiments[aspect] = "No sentiment found"  # Handle cases with no sentiments

    return aspect_sentiments

# Main function to select data source
def main():
    st.sidebar.title("Reddit Sentiment Analysis")
    
    subreddit = st.sidebar.text_input("Enter a subreddit:")
    num_posts = st.sidebar.number_input("Number of posts to analyze:", min_value=5, max_value=100, value=10)

    # Create tabs for navigation
    tab1, tab2 = st.tabs(["Sentiment Analysis", "Aspect Sentiment"])

    if st.sidebar.button("Analyze"):
        if subreddit:
            df = collect_reddit_data(subreddit, limit=num_posts)

            # Initialize the sentiment analysis pipeline
            sentiment_pipeline = pipeline("sentiment-analysis")

            # Sentiment Analysis
            df['sentiment'] = df['text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')  # Convert UTC to datetime

            # Create clickable links for the post titles
            df['Post'] = df.apply(lambda row: f'<a href="https://reddit.com{row["url"]}" target="_blank">{row["text"]}</a>', axis=1)

            # Add Serial Numbers
            df['#'] = range(1, len(df) + 1)

            # Add CSS for better styling with scrollable tables
            st.markdown("""
                <style>
                    .table-container {
                        max-height: 400px;
                        max-width: 100%;
                        overflow-y: auto;
                        overflow-x: auto;
                    }
                    table {
                        border-collapse: collapse;
                        margin: 25px 0;
                        font-size: 0.9em;
                        font-family: sans-serif;
                        min-width: 600px;  /* Increased width */
                        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                        width: 100%;
                    }
                    table thead {
                        position: sticky;
                        top: 0;
                        background-color: #FF4500;
                        z-index: 1;
                    }
                    table thead tr {
                        background-color: #FF4500;
                        color: #ffffff;
                        text-align: center;  /* Center align headers */
                    }
                    table th,
                    table td {
                        padding: 12px 15px;
                        white-space: nowrap;  /* Prevent text wrapping */
                        overflow: hidden;  /* Hide overflow */
                        text-overflow: ellipsis;  /* Show ellipsis */
                    }
                    table tbody tr {
                        border-bottom: 1px solid #dddddd;
                    }
                    table tbody tr:nth-of-type(even) {
                        background-color: #f3f3f3;
                    }
                    table tbody tr:last-of-type {
                        border-bottom: 2px solid #FF4500;
                    }
                    /* Specific width for the Post column */
                    table td:nth-child(2) {
                        max-width: 300px;  /* Adjust width for Post column */
                    }
                </style>
            """, unsafe_allow_html=True)

            # Wrap tables in scrollable containers
            # For Post Analysis table
            with tab1:
                st.write("### Post Analysis")
                st.markdown('<div class="table-container">' + 
                            df[['#', 'Post', 'author', 'score', 'sentiment', 'date']].to_html(escape=False, index=False) +
                            '</div>', 
                            unsafe_allow_html=True)

            # Sentiment Summary
            sentiment_summary = df['sentiment'].value_counts()
            st.write("### Sentiment Summary")
            st.write(sentiment_summary)

            # Extract aspects dynamically
            all_aspects = []
            for text in df['text']:
                aspects = extract_aspects(text)
                all_aspects.extend(aspects)

            # Remove duplicates and show in expander
            unique_aspects = list(set(all_aspects))
            with st.expander("### Extracted Aspects", expanded=False):
                st.write(unique_aspects)

            # Aspect-Based Sentiment Analysis
            df['aspect_sentiment'] = df['text'].apply(lambda x: aspect_based_sentiment(x, unique_aspects, sentiment_pipeline))

            # Display aspect sentiment results
            with tab2:
                st.write("### Aspect Sentiment Results")
                aspect_sentiment_df = pd.DataFrame(df['aspect_sentiment'].tolist(), index=df.index)

                # Add Serial Numbers to Aspect Sentiment Results
                aspect_sentiment_df.insert(0, '#', range(1, len(aspect_sentiment_df) + 1))

                # Use Streamlit's built-in table display with HTML
                st.markdown('<div class="table-container">' + 
                            aspect_sentiment_df.to_html(escape=False, index=False) +
                            '</div>', 
                            unsafe_allow_html=True)

            # Visualization
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            # Word Count Visualization
            st.subheader("Top 15 Word Count")
            word_counts = df['text'].str.split(expand=True).stack().value_counts().head(15)
            st.bar_chart(word_counts)

            # Word Clouds
            st.subheader("Positive Word Cloud")
            positive_words = ' '.join(df[df['sentiment'] == 'POSITIVE']['text'])
            positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
            st.image(positive_wordcloud.to_array(), use_column_width=True)

            st.subheader("Negative Word Cloud")
            negative_words = ' '.join(df[df['sentiment'] == 'NEGATIVE']['text'])
            negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_words)
            st.image(negative_wordcloud.to_array(), use_column_width=True)

            # Sentiment Over Time (Hourly)
            st.subheader("Sentiment Over Time (Hourly)")
            hourly_sentiment = df.groupby([df['date'].dt.floor('H'), 'sentiment']).size().unstack(fill_value=0)
            st.line_chart(hourly_sentiment)

            # Download Option
            st.download_button(
                label="Download Analyzed Data",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='reddit_sentiment_analysis.csv',
                mime='text/csv'
            )

        else:
            st.warning("Please enter a subreddit name.")

if __name__ == "__main__":
    main() 