import pandas as pd
import random
import numpy as np

def create_indian_spam_dataset(num_samples=2000):
    # Indian spam message templates
    spam_templates = [
        "Win ₹10000 cash prize! Call {number} to claim your reward. Hurry!",
        "You have won a free trip to Goa! Text YES to {number} to claim your package.",
        "Airtel: Your account has won a special bonus. Dial {number} to redeem now.",
        "Jio: Congratulations! You've been selected for 10GB free data. Claim at {number}",
        "Bank Alert: Your SBI card has won a lottery of ₹50000. Call {number} to process.",
        "URGENT: Your PAN card has been blocked. Verify immediately at {number}",
        "Flipkart Big Billion Days: You've won a smartphone! Claim at {number}",
        "Amazon India: You're a lucky winner of ₹20000 voucher. Call {number}",
        "Free recharge! Your number has been selected for ₹500 talktime. Dial {number}",
        "Aadhaar Alert: Your document needs verification. Contact {number} immediately.",
        "Your SIM has been deactivated. Reactivate now by calling {number}",
        "LIC Policy: You're eligible for bonus amount. Contact {number} to claim.",
        "Income Tax Refund: You have unclaimed refund of ₹12500. Call {number}",
        "Your WhatsApp account will be banned in 24 hours. Verify at {number}",
        "Paytm: Congratulations! You've won ₹1000 cashback. Claim at {number}"
    ]
    
    # Indian legitimate message templates
    ham_templates = [
        "Hi beta, when are you coming home? Mom has made your favorite food.",
        "Don't forget to bring milk on your way back from office.",
        "Meeting postponed to 3 PM tomorrow. Please update your calendar.",
        "Happy Diwali! May this festival bring joy and prosperity to your family.",
        "I'll be 15 minutes late for dinner. Traffic is bad on MG Road.",
        "Did you finish the project report? Need to submit it by EOD.",
        "Call me when you're free. Need to discuss something important.",
        "Remember to wish Papa happy birthday at midnight!",
        "Your parcel has been delivered. Check at the security desk.",
        "Let's meet for chai at the usual place at 5 PM?",
        "Can you pick up the kids from school today? I have a meeting.",
        "Don't forget to pay the electricity bill today to avoid late fee.",
        "Dinner at Saravana Bhavan tonight? My treat!",
        "Your doctor appointment is confirmed for tomorrow 11 AM.",
        "Movie tickets booked for Saturday evening. Show at 7 PM."
    ]
    
    # Generate Indian phone numbers
    def generate_indian_number():
        prefixes = ['+91', '91', '0']
        return random.choice(prefixes) + str(random.randint(7000000000, 9999999999))
    
    # Create dataset
    data = []
    labels = []
    
    # Generate spam messages
    for _ in range(num_samples // 2):
        template = random.choice(spam_templates)
        number = generate_indian_number()
        message = template.format(number=number)
        data.append(message)
        labels.append('spam')
    
    # Generate ham messages
    for _ in range(num_samples // 2):
        message = random.choice(ham_templates)
        data.append(message)
        labels.append('ham')
    
    # Shuffle the dataset
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    
    # Create DataFrame
    df = pd.DataFrame({'label': labels, 'message': data})
    return df

if __name__ == "__main__":
    # Create and save the Indian dataset
    df = create_indian_spam_dataset(2000)
    df.to_csv('data/indian_spam_dataset.csv', index=False)
    print("Indian spam dataset created with 2000 samples!")
    print(df['label'].value_counts())
