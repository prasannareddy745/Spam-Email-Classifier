import pandas as pd
import re
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

class SpamDetector:
    def __init__(self, dataset_type='standard'):
        self.ps = PorterStemmer()
        self.vectorizer = None
        self.model = None
        self.dataset_type = dataset_type
        
        # Set stop words based on dataset type
        self.stop_words = set(stopwords.words('english'))
        if dataset_type == 'indian':
            # Add common Indian words to stop words
            self.stop_words.update(['ji', 'aap', 'aapka', 'kyu', 'hai', 'main', 'mera', 'ko', 'ka', 'ki', 'se', 'mein'])
    
    def load_dataset(self):
        if self.dataset_type == 'standard':
            filepath = 'data/spam.csv'
            df = pd.read_csv(filepath, delimiter='\t', names=['label', 'message'])
        else:  # indian
            filepath = 'data/indian_spam_dataset.csv'
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Indian dataset not found at {filepath}. Please run dataset_generator.py first.")
            df = pd.read_csv(filepath)
        return df
    
    def preprocess_text(self, text):
        # Remove special characters and numbers but keep currency symbols
        if self.dataset_type == 'indian':
            text = re.sub(r'[^a-zA-ZÀ-ÿऀ-ॿ₹\s]', ' ', text)
        else:
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        words = [self.ps.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def train_model(self, test_size=0.2, vectorizer_type='tfidf'):
        # Load dataset
        df = self.load_dataset()
        
        # Preprocess messages
        print("Preprocessing messages...")
        df['processed'] = df['message'].apply(self.preprocess_text)
        
        # Prepare target variable
        y = df['label'].apply(lambda x: 1 if x == 'spam' else 0).values
        
        # Vectorize text
        print("Vectorizing text...")
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
            
        X = self.vectorizer.fit_transform(df['processed']).toarray()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        print("Training model...")
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with {accuracy:.4f} accuracy")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_test, y_test, df
    
    def predict_message(self, message):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        processed_msg = self.preprocess_text(message)
        vectorized_msg = self.vectorizer.transform([processed_msg]).toarray()
        prediction = self.model.predict(vectorized_msg)
        probability = self.model.predict_proba(vectorized_msg)
        
        return {
            'prediction': 'spam' if prediction[0] == 1 else 'ham',
            'spam_probability': probability[0][1],
            'ham_probability': probability[0][0]
        }
    
    def save_model(self):
        filename = f'models/{self.dataset_type}_spam_model.pkl'
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'stop_words': self.stop_words,
            'dataset_type': self.dataset_type
        }, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self):
        filename = f'models/{self.dataset_type}_spam_model.pkl'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found. Please train the model first.")
        
        data = joblib.load(filename)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.stop_words = data['stop_words']
        self.dataset_type = data['dataset_type']
        print(f"Model loaded from {filename}")

def interactive_test(detector):
    print(f"\nInteractive Testing ({detector.dataset_type} dataset)")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        message = input("\nEnter a message to classify: ")
        if message.lower() == 'quit':
            break
        
        try:
            result = detector.predict_message(message)
            print(f"Prediction: {result['prediction']}")
            print(f"Spam Confidence: {result['spam_probability']:.4f}")
            print(f"Ham Confidence: {result['ham_probability']:.4f}")
        except Exception as e:
            print(f"Error: {e}")

def test_from_file(detector, filename):
    print(f"\nTesting messages from {filename}:")
    print("-" * 50)
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            messages = file.readlines()
        
        for i, message in enumerate(messages, 1):
            message = message.strip()
            if message:
                result = detector.predict_message(message)
                print(f"{i}. {message}")
                print(f"   Classification: {result['prediction']}")
                print(f"   Spam Confidence: {result['spam_probability']:.4f}\n")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")

def quantitative_evaluation(detector):
    # Load dataset
    df = detector.load_dataset()
    
    # Preprocess and vectorize
    df['processed'] = df['message'].apply(detector.preprocess_text)
    X = detector.vectorizer.transform(df['processed']).toarray()
    y = df['label'].apply(lambda x: 1 if x == 'spam' else 0).values
    
    # Make predictions
    y_pred = detector.model.predict(X)
    y_pred_proba = detector.model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    auc_score = roc_auc_score(y, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {fscore:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {detector.dataset_type.capitalize()} Dataset')
    plt.savefig(f'confusion_matrix_{detector.dataset_type}.png')
    print(f"\nConfusion matrix saved as 'confusion_matrix_{detector.dataset_type}.png'")

def main():
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    while True:
        print("\n" + "="*50)
        print("UNIFIED SPAM DETECTION SYSTEM")
        print("="*50)
        print("1. Use Standard SMS Dataset")
        print("2. Use Indian SMS Dataset")
        print("3. Exit")
        
        dataset_choice = input("\nSelect dataset (1-3): ")
        
        if dataset_choice == '3':
            print("Exiting...")
            break
        
        if dataset_choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue
        
        dataset_type = 'standard' if dataset_choice == '1' else 'indian'
        detector = SpamDetector(dataset_type)
        
        # Inner menu for dataset-specific operations
        while True:
            print(f"\n{dataset_type.upper()} DATASET MENU")
            print("-" * 30)
            print("1. Train model")
            print("2. Load existing model")
            print("3. Interactive testing")
            print("4. Test from file")
            print("5. Quantitative evaluation")
            print("6. Save model")
            print("7. Back to main menu")
            
            operation_choice = input("\nSelect operation (1-7): ")
            
            if operation_choice == '7':
                break
                
            try:
                if operation_choice == '1':
                    test_size = float(input("Test size (0.1-0.4, default 0.2): ") or 0.2)
                    vectorizer_type = input("Vectorizer type (tfidf/count, default tfidf): ") or 'tfidf'
                    detector.train_model(test_size, vectorizer_type)
                    
                elif operation_choice == '2':
                    detector.load_model()
                    
                elif operation_choice == '3':
                    interactive_test(detector)
                    
                elif operation_choice == '4':
                    filename = input("Enter filename with messages: ")
                    test_from_file(detector, filename)
                    
                elif operation_choice == '5':
                    quantitative_evaluation(detector)
                    
                elif operation_choice == '6':
                    detector.save_model()
                    
                else:
                    print("Invalid choice. Please try again.")
                    
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
