import os
import json
import requests
import zipfile
import csv


def download_file(url, save_path, extract=False, extracted_dir=None):
    """
    Download a file from a URL if it doesn't already exist.
    Optionally extract the file if it's a ZIP and skip extraction
    if the extracted directory already exists.
    
    Args:
        url (str): URL to download the file from.
        save_path (str): Path to save the downloaded file.
        extract (bool): Whether to extract the file if it's a ZIP.
        extracted_dir (str): Path to the extracted directory to check if already extracted.
    """
    # Ensure the directory for save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Check if the file already exists
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}. Skipping download.")
    else:
        # Download the file
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {save_path}")
    
    # Check if extraction is needed
    if extract:
        if extracted_dir and os.path.exists(extracted_dir):
            print(f"Extracted data already exists at {extracted_dir}. Skipping extraction.")
        else:
            print(f"Extracting {save_path}...")
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(save_path))
            print(f"Extraction complete. Data extracted to {os.path.dirname(save_path)}")




def process_glove(glove_path, output_json, dataset_vocab=None, embedding_dim=300):
    """Process GloVe file and save vocabulary and embeddings to JSON."""
    print("Processing GloVe file...")
    vocab = []
    embeddings = []
    with open(glove_path, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            if dataset_vocab is None or word in dataset_vocab:
                vocab.append(word)
                embeddings.append([float(x) for x in split_line[1:]])

    # Add <UNK> token with zero embedding
    vocab.append("<UNK>")
    embeddings.append([0.0] * embedding_dim)

    # Save to JSON
    data = {'vocab': vocab, 'emba': embeddings}
    with open(output_json, 'w') as json_file:
        json.dump(data, json_file)

    print(f"Vocabulary size (excluding <UNK>): {len(vocab) - 1}")
    print(f"Vocabulary size (including <UNK>): {len(vocab)}")
    print(f"Processed GloVe embeddings saved to {output_json}")

def extract_dataset_vocab(dataset_path):
    """Extract unique words from the Sentiment140 dataset."""
    print("Extracting vocabulary from dataset...")
    vocab = set()
    with open(dataset_path, 'r', encoding="ISO-8859-1") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            message = row[-1]  # The message is the last column
            words = message.lower().split()  # Tokenize by splitting on spaces
            vocab.update(words)
    print(f"Extracted {len(vocab)} unique words from the dataset.")
    return vocab

def main():
    # URLs and paths
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    glove_dir = "./glove"
    glove_file = "glove.6B.300d.txt"
    dataset_url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    dataset_dir = "./sentiment140"
    train_file = os.path.join(dataset_dir, "training.1600000.processed.noemoticon.csv")
    output_json = "embs.json"
    embedding_dim = 300

    # Download and extract GloVe embeddings
    download_file(glove_url, os.path.join(glove_dir, 'glove.6B.zip'), extract=True)

    # Download and extract Sentiment140 dataset
    download_file(dataset_url, os.path.join(dataset_dir, 'trainingandtestdata.zip'), extract=True)

    # Extract dataset vocabulary
    dataset_vocab = extract_dataset_vocab(train_file)

    # Process GloVe and save embeddings
    process_glove(os.path.join(glove_dir, glove_file), output_json, dataset_vocab, embedding_dim)

if __name__ == "__main__":
    main()
