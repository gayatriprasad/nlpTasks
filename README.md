# NLP Toolkit

This project contains a set of unified scripts for various Natural Language Processing (NLP) tasks. It provides easy-to-use interfaces for language detection, sentiment analysis, entity recognition, and key phrase extraction.

## Scripts

1. `language_detection.py`: Detects the language of given text using multiple methods.
2. `sentiment_analysis.py`: Analyzes the sentiment of given text using various approaches.
3. `entity_recognition.py`: Extracts named entities from text using different techniques.
4. `keyphrase_extraction.py`: Extracts key phrases from text using multiple algorithms.

## Features

- Multiple methods available for each task, ranging from simple to advanced approaches.
- Includes integration with popular NLP libraries and LLM capabilities.
- User-friendly command-line interfaces for easy interaction.

## Requirements

- Python 3.7+
- Various Python libraries (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/nlp-toolkit.git
   cd nlp-toolkit
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - For OpenAI GPT integration, set your API key in each script or as an environment variable.

## Usage

Run each script from the command line:

```
python language_detection.py
python sentiment_analysis.py
python entity_recognition.py
python keyphrase_extraction.py
```

Follow the prompts to input text and choose analysis methods.

## Note

Some methods require additional model downloads or may not be available if certain libraries are not installed. The scripts will notify you of any missing components.