import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
packages = [
    "pandas==1.5.3",
    "numpy==1.24.3",
    "scikit-learn==1.2.2",
    "nltk==3.8.1",
    "joblib==1.2.0",
    "matplotlib==3.7.1",
    "seaborn==0.12.2"
]

for package in packages:
    try:
        install(package)
        print(f"Successfully installed {package}")
    except Exception as e:
        print(f"Error installing {package}: {e}")

# Download NLTK data
import nltk
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

print("Setup completed!")
