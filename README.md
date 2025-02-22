# Reddit Sentiment Analysis

## Overview

The **Reddit Sentiment Analysis** project is a powerful tool designed to analyze sentiments expressed in Reddit posts. By leveraging the capabilities of the Hugging Face Transformers library, this application performs both general sentiment analysis and aspect-based sentiment analysis, allowing users to gain insights into specific topics discussed within a subreddit.

## Features

- **Data Collection**: Fetches posts from any specified subreddit using the Reddit API.
- **Sentiment Analysis**: Analyzes the overall sentiment of each post (Positive, Negative, Neutral).
- **Aspect-Based Sentiment Analysis**: Extracts keywords and analyzes sentiments related to specific aspects mentioned in the posts.
- **Visualizations**: Displays sentiment distribution, word counts, and generates word clouds for positive and negative sentiments.
- **Downloadable Reports**: Users can download the analyzed data in CSV format for further analysis.

## Technologies Used

- **Python**: The primary programming language.
- **Streamlit**: For creating the web application interface.
- **Pandas**: For data manipulation and analysis.
- **Transformers**: For sentiment analysis using pre-trained models.
- **PRAW**: Python Reddit API Wrapper for accessing Reddit data.
- **NLTK**: Natural Language Toolkit for text processing.
- **RAKE**: Rapid Automatic Keyword Extraction for aspect extraction.
- **WordCloud**: For generating word clouds from text data.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/reddit-sentiment-analysis.git
   cd reddit-sentiment-analysis
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Reddit API credentials:
   ```plaintext
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

5. **Download NLTK resources**:
   The application will automatically download necessary NLTK resources when run for the first time.

## Usage

1. **Run the application**:
   ```bash
   streamlit run sentiment_analysis.py
   ```

2. **Access the application**:
   Open your web browser and go to `http://localhost:8501`.

3. **Analyze Sentiments**:
   - Enter a subreddit name in the input field.
   - Specify the number of posts to analyze.
   - Click the "Analyze" button to fetch and analyze the data.

4. **View Results**:
   - Navigate through the tabs to view post analysis, sentiment summaries, extracted aspects, and aspect sentiment results.
   - Visualizations will provide insights into sentiment distribution and word counts.
   - Download the analyzed data as a CSV file.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pre-trained models for sentiment analysis.
- [PRAW](https://praw.readthedocs.io/en/latest/) for simplifying Reddit API interactions.
- [Streamlit](https://streamlit.io/) for making it easy to create web applications with Python.

## Contact

For any inquiries, please reach out to [your.email@example.com](mailto:your.email@example.com).
