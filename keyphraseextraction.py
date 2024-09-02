import nltk
from rake_nltk import Rake
from summa import keywords
import yake
import openai

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

def extract_keyphrases_rake(text, num_phrases=5):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return "RAKE", rake.get_ranked_phrases()[:num_phrases]

def extract_keyphrases_textrank(text, num_phrases=5):
    result = keywords.keywords(text, words=num_phrases)
    return "TextRank", result.split('\n')

def extract_keyphrases_yake(text, num_phrases=5):
    kw_extractor = yake.KeywordExtractor(top=num_phrases, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, features=None)
    keywords = kw_extractor.extract_keywords(text)
    return "YAKE", [kw[0] for kw in keywords]

def extract_keyphrases_llm(text, num_phrases=5):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a key phrase extraction expert. Extract the top {num_phrases} key phrases from the given text. Respond with one key phrase per line, nothing else."},
            {"role": "user", "content": f"Extract key phrases from this text: '{text}'"}
        ],
        max_tokens=100
    )
    result = response.choices[0].message['content'].strip().split('\n')
    return "LLM (GPT-3.5)", result[:num_phrases]

def extract_keyphrases(text, method='rake', num_phrases=5):
    methods = {
        'rake': extract_keyphrases_rake,
        'textrank': extract_keyphrases_textrank,
        'yake': extract_keyphrases_yake,
        'llm': extract_keyphrases_llm
    }
    
    if method not in methods:
        raise ValueError(f"Invalid method. Choose from: {', '.join(methods.keys())}")
    
    method_name, keyphrases = methods[method](text, num_phrases)
    return method_name, keyphrases

def main():
    print("Welcome to the Unified Key Phrase Extraction Script!")
    
    while True:
        text = input("\nEnter the text for key phrase extraction (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        
        num_phrases = int(input("Enter the number of key phrases to extract: "))
        
        print("\nChoose a key phrase extraction method:")
        print("1. RAKE (Rapid Automatic Keyword Extraction)")
        print("2. TextRank")
        print("3. YAKE (Yet Another Keyword Extractor)")
        print("4. LLM (GPT-3.5)")
        
        choice = input("Enter your choice (1-4): ")
        
        method_map = {'1': 'rake', '2': 'textrank', '3': 'yake', '4': 'llm'}
        method = method_map.get(choice, 'rake')
        
        try:
            method_name, keyphrases = extract_keyphrases(text, method, num_phrases)
            print(f"\nMethod used: {method_name}")
            print("Key phrases extracted:")
            for i, keyphrase in enumerate(keyphrases, 1):
                print(f"{i}. {keyphrase}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()