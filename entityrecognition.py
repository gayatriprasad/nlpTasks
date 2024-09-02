import spacy
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from transformers import pipeline
import openai

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    print("spaCy model is not available. The spaCy method will be disabled.")

# Set up Hugging Face pipeline
try:
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("Hugging Face Transformers is not available. The Transformers method will be disabled.")

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

def extract_entities_spacy(text):
    if not SPACY_AVAILABLE:
        return "spaCy not available", []
    
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return "spaCy", entities

def extract_entities_nltk(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    entities = []
    for chunk in chunked:
        if isinstance(chunk, Tree):
            entities.append((' '.join(c[0] for c in chunk.leaves()), chunk.label()))
    return "NLTK", entities

def extract_entities_transformers(text):
    if not TRANSFORMERS_AVAILABLE:
        return "Transformers not available", []
    
    results = ner_pipeline(text)
    entities = [(entity['word'], entity['entity_group']) for entity in results]
    return "Hugging Face Transformers", entities

def extract_entities_llm(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an entity recognition expert. Identify entities in the given text and return them in the format: Entity: Type. Separate each entity with a newline."},
            {"role": "user", "content": f"Extract entities from this text: '{text}'"}
        ],
        max_tokens=150
    )
    result = response.choices[0].message['content'].strip().split('\n')
    entities = [tuple(entity.split(': ')) for entity in result if ': ' in entity]
    return "LLM (GPT-3.5)", entities

def extract_entities(text, method='spacy'):
    methods = {
        'spacy': extract_entities_spacy,
        'nltk': extract_entities_nltk,
        'transformers': extract_entities_transformers,
        'llm': extract_entities_llm
    }
    
    if method not in methods:
        raise ValueError(f"Invalid method. Choose from: {', '.join(methods.keys())}")
    
    method_name, entities = methods[method](text)
    return method_name, entities

def main():
    print("Welcome to the Unified Entity Recognition Script!")
    
    while True:
        text = input("\nEnter the text for entity recognition (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        
        print("\nChoose an entity recognition method:")
        if SPACY_AVAILABLE:
            print("1. spaCy")
        print("2. NLTK")
        if TRANSFORMERS_AVAILABLE:
            print("3. Hugging Face Transformers")
        print("4. LLM (GPT-3.5)")
        
        choice = input("Enter your choice: ")
        
        method_map = {'1': 'spacy', '2': 'nltk', '3': 'transformers', '4': 'llm'}
        method = method_map.get(choice, 'nltk')
        
        if method == 'spacy' and not SPACY_AVAILABLE:
            print("spaCy is not available. Please choose another method.")
            continue
        if method == 'transformers' and not TRANSFORMERS_AVAILABLE:
            print("Hugging Face Transformers is not available. Please choose another method.")
            continue
        
        try:
            method_name, entities = extract_entities(text, method)
            print(f"\nMethod used: {method_name}")
            print("Entities found:")
            for entity, entity_type in entities:
                print(f"- {entity}: {entity_type}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()