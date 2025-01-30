# Sentiment140 Dataset Preprocessor

## Overview

This script processes the Sentiment140 dataset and GloVe embeddings to create a combined JSON file for sentiment analysis tasks. It downloads GloVe embeddings automatically and processes them along with the Sentiment140 dataset to create word embeddings specifically tailored for sentiment analysis.



## Dataset Setup

1. Download the Sentiment140 dataset:
   * Visit [Hugging Face Sentiment140](https://huggingface.co/datasets/stanfordnlp/sentiment140)
   * Download the dataset ZIP file
   * Create a directory named `sentiment140` in this folder
   * Extract the ZIP contents into the `sentiment140` directory

2. Directory structure should look like:
   ```
   ./
   ├── sentiment140/
   │   └── training.1600000.processed.noemoticon.csv
   ├── glove/
   │   └── (will be created automatically)
   └── process_embeddings.py
   ```

## Running the Script

Simply run:
```bash
python process_embeddings.py
```

The script will:
1. Automatically download GloVe embeddings (glove.6B)
2. Extract vocabulary from Sentiment140 dataset
3. Process GloVe embeddings for the dataset vocabulary
4. Create `embs.json` containing the processed embeddings

## Output

The script generates `embs.json` with the following structure:
```json
{
    "vocab": ["word1", "word2", ..., "<UNK>"],
    "emba": [[embedding1], [embedding2], ..., [unk_embedding]]
}
```

* Vocabulary includes all unique words from the dataset
* Each word has a 300-dimensional embedding vector
* Includes `<UNK>` token for unknown words
* Embeddings are filtered to only include words present in the dataset


## Notes

* The script uses GloVe 6B embeddings with 300 dimensions
* Processing may take several minutes depending on your system
* The output JSON file may be large due to the dataset size

