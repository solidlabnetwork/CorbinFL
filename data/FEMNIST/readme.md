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

## JSON File Format

Each JSON file contains data for a subset of writers in a structured format. Below is a detailed description of the structure:

### JSON Structure
```json
{
  "users": ["writer1", "writer2", ...],
  "num_samples": [num_samples1, num_samples2, ...],
  "user_data": {
    "writer1": {
      "x": [
        [flattened_pixel_values_1],
        [flattened_pixel_values_2],
        ...
      ],
      "y": [
        label_1,
        label_2,
        ...
      ]
    },
    "writer2": {
      "x": [
        [flattened_pixel_values_1],
        [flattened_pixel_values_2],
        ...
      ],
      "y": [
        label_1,
        label_2,
        ...
      ]
    }
  }
}
```

### Fields in the JSON File

1. **`users`**
   - A list of writer IDs included in the file.

2. **`num_samples`**
   - A list of integers representing the number of samples (images) contributed by each writer.

3. **`user_data`**
   - A dictionary where each key is a writer ID, and the value is another dictionary containing:
     - **`x`**: A list of flattened and normalized pixel values for the writer's images.
       - Each image is resized to `28x28` pixels, converted to grayscale, flattened into a vector of length `784`, and normalized to values between `0.0` and `1.0`.
     - **`y`**: A list of integer labels corresponding to the images in `x`.
       - Labels are relabeled into a compact numeric range (`0-61`) to represent digits (`0-9`), uppercase letters (`10-35`), and lowercase letters (`36-61`).

### Example JSON File Content
```json
{
  "users": ["writer_001", "writer_002"],
  "num_samples": [2, 3],
  "user_data": {
    "writer_001": {
      "x": [
        [0.0, 0.1, 0.2, ..., 0.3],
        [0.4, 0.5, 0.6, ..., 0.7]
      ],
      "y": [10, 15]
    },
    "writer_002": {
      "x": [
        [0.1, 0.1, 0.1, ..., 0.2],
        [0.2, 0.3, 0.4, ..., 0.5],
        [0.3, 0.4, 0.5, ..., 0.6]
      ],
      "y": [5, 20, 25]
    }
  }
}
```

---

## Key Details

- **Images:** Each image is resized to `28x28`, converted to grayscale, flattened, and normalized to values between `0.0` and `1.0`.
- **Labels:** Relabeled to integers in the range `0-61` for consistency across all data.
- **Batched Writers:** To manage file size, data for a limited number of writers (based on `max_writers_per_file`) is stored in each JSON file.
- **Purpose:** This structure ensures compatibility with machine learning models and facilitates batch processing during training.

---

## Summary

This preprocessing pipeline organizes and converts the FEMNIST dataset into a machine-readable JSON format, with clear grouping by writer and compact relabeling of class labels. The resulting data is stored in an efficient and scalable structure suitable for training federated learning models.





