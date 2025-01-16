
"""
Vocabulary builder for Reddit dataset.
This script processes JSON files in the format:
{
    'user_data': {
        'user_id': {
            'x': [list_of_sequences]
        }
    }
}
and creates a vocabulary file (reddit_vocab.pck) containing:
{
    'vocab': {word: index},
    'size': vocab_size,
    'pad_symbol': 0,
    'unk_symbol': 1
}
How to run:
python Reddit_vocab_creation.py --vocab-size 10000 --data-dir reddit_leaf/train
"""

import argparse
import collections
import json
import os
import pickle
from typing import Dict, Counter

def load_json_data(file_path: str) -> Dict:
    """Load user data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing user data
    """
    print(f'Loading {os.path.basename(file_path)}')
    with open(file_path, 'r') as f:
        data = json.load(f)
        user_data = data['user_data']
    return user_data

def build_counter(train_data: Dict, initial_counter: Counter = None) -> Counter:
    """Build a token counter from training data.
    
    Args:
        train_data: Dictionary of user data
        initial_counter: Existing counter to update (optional)
        
    Returns:
        Counter object with token frequencies
    """
    # Initialize or use existing counter
    counter = collections.Counter() if initial_counter is None else initial_counter
    
    # Process nested data structure
    for user_id in train_data:
        for sequence_list in train_data[user_id]['x']:
            # Handle case where tokens are in nested lists
            if isinstance(sequence_list[0], list):
                # Flatten the nested list structure
                for token_list in sequence_list:
                    counter.update(token_list)
            else:
                # Direct count if tokens are not nested
                counter.update(sequence_list)
                
    return counter

def build_vocab(counter: Counter, vocab_size: int) -> Dict:
    """Build vocabulary from token counter.
    
    Args:
        counter: Counter object with token frequencies
        vocab_size: Desired size of vocabulary
        
    Returns:
        Dictionary containing vocabulary and metadata
    """
    # Reserve 0 for PAD and 1 for UNK
    pad_symbol, unk_symbol = 0, 1
    
    # Sort tokens by frequency (descending) and alphabetically for ties
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    # Select top tokens (leaving room for PAD and UNK)
    count_pairs = count_pairs[:(vocab_size - 1)]
    
    # Extract words (discarding counts)
    words = [word for word, _ in count_pairs]
    
    # Create vocabulary dictionary
    vocab = {
        '<PAD>': pad_symbol,  # Padding token
        '<UNK>': unk_symbol,  # Unknown token
    }
    
    # Add words with indices starting after PAD and UNK
    current_idx = 2  # Start from 2 after PAD and UNK
    for word, _ in count_pairs:
        if word not in ['<PAD>', '<UNK>']:  # Skip both special tokens
            vocab[word] = current_idx
            current_idx += 1

    final_size = len(vocab)
     
    return {
        'vocab': vocab,
        'size': final_size,
        'pad_symbol': pad_symbol,
        'unk_symbol': unk_symbol
    }

def save_vocab(vocab: Dict, target_dir: str):
    """Save vocabulary to pickle file.
    
    Args:
        vocab: Vocabulary dictionary
        target_dir: Directory to save vocabulary file
    """
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, 'reddit_vocab.pck')
    
    with open(output_path, 'wb') as f:
        pickle.dump(vocab, f)
        
    print(f'\nVocabulary saved to {output_path}')
    print(f'Vocabulary size: {len(vocab["vocab"])}')
    print(f'Sample tokens: {list(vocab["vocab"].items())[:10]}')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build vocabulary from Reddit dataset')
    
    parser.add_argument(
        '--data-dir',
        help='Directory containing training JSON files',
        type=str,
        required=True
    )
    
    parser.add_argument(
        '--vocab-size',
        help='Size of vocabulary (including PAD and UNK tokens)',
        type=int,
        default=50000,
        required=False
    )
    
    parser.add_argument(
        '--target-dir',
        help='Directory to save vocabulary file',
        type=str,
        default='./',
        required=False
    )
    
    args = parser.parse_args()
    
    # Process all JSON files in data directory
    counter = None
    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    json_files.sort()  # Ensure consistent ordering
    
    print(f'\nProcessing {len(json_files)} files from {args.data_dir}')
    
    for json_file in json_files:
        file_path = os.path.join(args.data_dir, json_file)
        train_data = load_json_data(file_path)
        print(f'Counting tokens in {json_file}')
        counter = build_counter(train_data, initial_counter=counter)
    
    if counter is not None:
        print(f'\nTotal unique tokens found: {len(counter)}')
        print(f'Creating vocabulary with size {args.vocab_size}')
        
        # Build and save vocabulary
        vocab = build_vocab(counter, vocab_size=args.vocab_size)
        save_vocab(vocab, args.target_dir)
    else:
        print('No files to process.')

if __name__ == '__main__':
    main()