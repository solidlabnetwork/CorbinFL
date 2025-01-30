# FEMNIST Dataset Preprocessing and Storage

This document outlines the structure and format of the FEMNIST dataset as processed by the provided script. It describes how the data is stored and how the JSON files are organized.

---

## Quick Start

```bash
# Run with default settings (uses current directory)
python femnist_preprocessor.py

# Or specify custom directory
python femnist_preprocessor.py --data_dir /path/to/your/directory
```

## Directory Structure

```
data_dir/
├── raw_data/          # Raw datasets
│   ├── by_class/      # Class-organized images
│   └── by_write/      # Writer-organized images
├── intermediate/      # Processing checkpoints
└── all_data/         # Processed JSONs
```



### 2. **`intermediate/`**
- Stores intermediate processing results in serialized Python objects using `pickle`. These include:
  - **`class_file_dirs`**: List of file paths for images grouped by class.
  - **`write_file_dirs`**: List of file paths for images grouped by writer.
  - **`class_file_hashes`**: Hashes of files grouped by class.
  - **`write_file_hashes`**: Hashes of files grouped by writer.
  - **`write_with_class`**: Mapped writer-class pairs (with hashes linked).

### 3. **`all_data/`**
- Stores the final processed dataset in JSON format.
- Files are named sequentially as `all_data_0.json`, `all_data_1.json`, etc., based on the number of writers and the `max_writers_per_file` parameter.

---


## Key Details

- **Images:** Each image is resized to `28x28`, converted to grayscale, flattened, and normalized to values between `0.0` and `1.0`.
- **Labels:** Relabeled to integers in the range `0-61` for consistency across all data.
- **Batched Writers:** To manage file size, data for a limited number of writers (based on `max_writers_per_file`) is stored in each JSON file.
- **Purpose:** This structure ensures compatibility with machine learning models and facilitates batch processing during training.

---

## Summary

This preprocessing pipeline organizes and converts the FEMNIST dataset into a machine-readable JSON format, with clear grouping by writer and compact relabeling of class labels. The resulting data is stored in an efficient and scalable structure suitable for training federated learning models.





