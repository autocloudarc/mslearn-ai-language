"""
Text Analysis Script

This script analyzes customer reviews using Azure AI Language Services.
It performs the following text analysis operations:
- Language detection
- Sentiment analysis
- Key phrase extraction
- Named entity recognition
- Linked entity recognition

The script reads review files from a local 'reviews' folder and outputs
analysis results to the console.
"""

# Load environment variables from .env file (contains API credentials)
from dotenv import load_dotenv
import os

# Azure authentication modules
from azure.core.credentials import AzureKeyCredential  # For API key-based authentication
from azure.identity import DefaultAzureCredential  # For managed identity/EntraID authentication

# Azure Text Analytics modules for NLP operations
from azure.ai.textanalytics import TextAnalyticsClient  # Main client for text analysis
from azure.ai.textanalytics import DetectLanguageInput, TextDocumentInput  # Input models for analysis
from azure.core.exceptions import HttpResponseError  # Exception handling for Azure API errors


def main():
    """
    Main function to orchestrate text analysis of review documents.
    Reads configuration, authenticates with Azure, and analyzes each review file.
    """
    try:
        # Load configuration settings from environment variables (.env file)
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')  # Azure service endpoint URL
        
        # Create authentication credential using DefaultAzureCredential
        # This automatically tries multiple authentication methods in order:
        # 1. Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
        # 2. Managed Identity (if running in Azure)
        # 3. Visual Studio credentials
        # 4. Azure CLI credentials
        # 5. Interactive login
        credential = DefaultAzureCredential()
        
        # Instantiate TextAnalyticsClient with the endpoint and authentication credential
        # This client will be used to make all text analysis API calls to Azure
        client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)


        # Process all review files in the 'reviews' directory
        reviews_folder = 'reviews'
        for file_name in os.listdir(reviews_folder):
            # Print separator and file name for readability
            print('\n-------------\n' + file_name)
            
            # Read the complete contents of each review file
            # Using utf8 encoding to properly handle special characters
            text = open(os.path.join(reviews_folder, file_name), encoding='utf8').read()
            print('\n' + text)

            # LANGUAGE DETECTION: Identify the primary language of the review
            # Returns language name and ISO 639-1 code (e.g., 'en', 'es')
            detected_language = client.detect_language(documents=[text])[0]
            print(f'\nLanguage: {detected_language.primary_language.name}')

            # SENTIMENT ANALYSIS: Determine the overall sentiment (positive, negative, or neutral)
            # Analyzes the emotional tone and provides confidence scores (0-1) for each sentiment
            sentiment_analysis = client.analyze_sentiment(documents=[text], language=detected_language.primary_language.iso6391_name)[0]
            print(f'\nSentiment: {sentiment_analysis.sentiment}')
            # Display confidence scores for all three sentiment categories (formatted to 2 decimal places)
            print(f'Scores: Positive={sentiment_analysis.confidence_scores.positive:.2f}, Negative={sentiment_analysis.confidence_scores.negative:.2f}, Neutral={sentiment_analysis.confidence_scores.neutral:.2f}')

            # KEY PHRASE EXTRACTION: Identify the most important phrases/topics in the text
            # These are typically nouns or noun phrases that represent main concepts
            phrases = client.extract_key_phrases(documents=[text], language=detected_language.primary_language.iso6391_name)[0]
            if phrases.key_phrases:
                print('\nKey Phrases:')
                for phrase in phrases.key_phrases:
                    print(f'\t{phrase}')

            # NAMED ENTITY RECOGNITION: Extract specific entities like persons, organizations, locations, etc.
            # Categorizes identified entities into predefined types
            entities = client.recognize_entities(documents=[text], language=detected_language.primary_language.iso6391_name)[0]
            if entities.entities:
                print('\nEntities:')
                # Display each identified entity with its category (e.g., Person, Organization, Location)
                for entity in entities.entities:
                    print(f'\t{entity.text} ({entity.category})')

            # LINKED ENTITY RECOGNITION: Identify entities and link them to Wikipedia entries
            # Provides semantic linking to known entities for richer context and information retrieval
            linked_entities = client.recognize_linked_entities(documents=[text], language=detected_language.primary_language.iso6391_name)[0]
            if linked_entities.entities:
                print('\nLinked Entities:')
                # Display entity names with their corresponding Wikipedia URLs
                for entity in linked_entities.entities:
                    print(f'\t{entity.name}: {entity.url}')



    except Exception as ex:
        # Handle any errors during execution (authentication, API calls, file I/O, etc.)
        print(ex)


if __name__ == "__main__":
    # Entry point: Run the main function when script is executed directly
    main()