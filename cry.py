import csv
import random
import string
from cryptography.fernet import Fernet

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def complete_columnar_transposition(text, period):
    # Pad text to make it divisible by period
    padded_text = text + ' ' * (period - (len(text) % period))
    
    # Create grid
    grid = [list(padded_text[i:i+period]) for i in range(0, len(padded_text), period)]
    
    # Randomly select columns order
    columns = list(range(period))
    random.shuffle(columns)
    
    # Reconstruct encrypted text
    encrypted = ''.join([''.join(row[col] for row in grid) for col in columns])
    return encrypted

def compressocrat(text):
    # Simple substitution-based encryption
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[3:] + alphabet[:3]
    trans_table = str.maketrans(alphabet, shifted_alphabet)
    return text.lower().translate(trans_table)

def generate_encryption_dataset(num_entries=400):
    dataset = []
    
    for _ in range(num_entries):
        # Generate random plaintext
        plaintext = generate_random_string(random.randint(110, 150))
        
        # Randomly choose encryption algorithm
        encryption_methods = [
            ('Complete Columnar Transposition', 
             complete_columnar_transposition(plaintext, random.randint(8, 15))),
            ('Compressocrat', 
             compressocrat(plaintext))
        ]
        
        method_name, ciphertext = random.choice(encryption_methods)
        
        dataset.append([plaintext, ciphertext, method_name])
    
    return dataset

def save_dataset_to_csv(dataset, filename='encryption_dataset.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Plaintext', 'Ciphertext', 'Algorithm'])
        writer.writerows(dataset)

# Generate and save dataset
random.seed(42)  # For reproducibility
dataset = generate_encryption_dataset()
save_dataset_to_csv(dataset)
print(f"Dataset generated with {len(dataset)} entries.")