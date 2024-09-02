import langdetect
import langid
import wget
import os
import openai

# Ensure you have the necessary libraries installed:
# pip install langdetect langid fasttext wget openai

# Attempt to import FastText, but don't fail if it's not available
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("FastText is not available. The FastText method will be disabled.")

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

def detect_language_simple(text):
    try:
        return langdetect.detect(text), None  # No confidence score available
    except:
        return "Unable to detect language", None

def detect_language_langid(text):
    lang, confidence = langid.classify(text)
    return lang, confidence

def download_fasttext_model():
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    if not os.path.exists("lid.176.bin"):
        print("Downloading FastText model...")
        wget.download(url)
        print("\nDownload complete.")

def detect_language_fasttext(text):
    if not os.path.exists("lid.176.bin"):
        download_fasttext_model()
    
    model = fasttext.load_model("lid.176.bin")
    predictions = model.predict(text, k=1)  # top 1 prediction
    lang = predictions[0][0].split('__')[-1]
    confidence = predictions[1][0]
    return lang, confidence

def detect_language_llm(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a language detection expert. Respond only with the ISO 639-1 code of the detected language."},
            {"role": "user", "content": f"Detect the language of this text: '{text}'"}
        ],
        max_tokens=10
    )
    return response.choices[0].message['content'].strip(), None  # No confidence score available

def detect_language(text, method='simple'):
    methods = {
        'simple': detect_language_simple,
        'langid': detect_language_langid,
        'fasttext': detect_language_fasttext,
        'llm': detect_language_llm
    }
    
    if method not in methods:
        raise ValueError(f"Invalid method. Choose from: {', '.join(methods.keys())}")
    
    lang, confidence = methods[method](text)
    return lang, confidence

def main():
    print("Welcome to the Unified Language Detection Script!")
    
    while True:
        text = input("\nEnter the text for language detection (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        
        print("\nChoose a detection method:")
        print("1. Simple (langdetect)")
        print("2. Langid")
        print("3. FastText")
        print("4. LLM (GPT-3.5)")
        
        choice = input("Enter your choice (1-4): ")
        
        method_map = {'1': 'simple', '2': 'langid', '3': 'fasttext', '4': 'llm'}
        method = method_map.get(choice, 'simple')
        
        try:
            lang, confidence = detect_language(text, method)
            print(f"\nDetected Language: {lang}")
            if confidence is not None:
                print(f"Confidence: {confidence}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()