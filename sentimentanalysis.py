import openai
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Attempt to import and set up transformers, but don't fail if it's not available
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("Hugging Face Transformers is not available. The Transformers method will be disabled.")

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive", compound
    elif compound <= -0.05:
        return "Negative", compound
    else:
        return "Neutral", compound

def analyze_sentiment_transformers(text):
    if not TRANSFORMERS_AVAILABLE:
        return "Transformers not available", None
    
    result = sentiment_pipeline(text)[0]
    
    label = result['label']
    score = result['score']
    
    if label == "POSITIVE":
        return "Positive", score
    elif label == "NEGATIVE":
        return "Negative", score
    else:
        return "Neutral", score

def analyze_sentiment_llm(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis expert. Respond with the sentiment (Positive, Negative, or Neutral) followed by a confidence score between 0 and 1, separated by a comma."},
            {"role": "user", "content": f"Analyze the sentiment of this text: '{text}'"}
        ],
        max_tokens=10
    )
    result = response.choices[0].message['content'].strip().split(',')
    sentiment, score = result[0].strip(), float(result[1].strip())
    return sentiment, score

def analyze_sentiment(text, method='textblob'):
    methods = {
        'textblob': analyze_sentiment_textblob,
        'vader': analyze_sentiment_vader,
        'transformers': analyze_sentiment_transformers,
        'llm': analyze_sentiment_llm
    }
    
    if method not in methods:
        raise ValueError(f"Invalid method. Choose from: {', '.join(methods.keys())}")
    
    sentiment, score = methods[method](text)
    return sentiment, score

def main():
    print("Welcome to the Unified Sentiment Analysis Script!")
    
    while True:
        text = input("\nEnter the text for sentiment analysis (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        
        print("\nChoose a sentiment analysis method:")
        print("1. TextBlob")
        print("2. VADER")
        if TRANSFORMERS_AVAILABLE:
            print("3. Hugging Face Transformers")
        print("4. LLM (GPT-3.5)")
        
        choice = input(f"Enter your choice (1-{'4' if TRANSFORMERS_AVAILABLE else '3'}, excluding Transformers if not available): ")
        
        method_map = {'1': 'textblob', '2': 'vader', '3': 'transformers', '4': 'llm'}
        method = method_map.get(choice, 'textblob')
        
        if method == 'transformers' and not TRANSFORMERS_AVAILABLE:
            print("Hugging Face Transformers is not available. Please choose another method.")
            continue
        
        try:
            sentiment, score = analyze_sentiment(text, method)
            print(f"\nSentiment: {sentiment}")
            if score is not None:
                print(f"Score/Confidence: {score}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()