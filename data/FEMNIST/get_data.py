import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import shutil
import hashlib
import json
import numpy as np
from PIL import Image
import pickle

def save_obj(obj, path):
    """Save a Python object using pickle"""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    """Load a Python object using pickle"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_file_dirs(parent_path):
    """Get file directories for class and write data"""
    class_files = []  # (class, file directory)
    write_files = []  # (writer, file directory)

    # Process by_class directory
    class_dir = os.path.join(parent_path, 'raw_data', 'by_class')
    rel_class_dir = os.path.join('raw_data', 'by_class')
    classes = [c for c in os.listdir(class_dir) if len(c) == 2]

    for cl in classes:
        cldir = os.path.join(class_dir, cl)
        rel_cldir = os.path.join(rel_class_dir, cl)
        subcls = [s for s in os.listdir(cldir) if (('hsf' in s) and ('mit' not in s))]

        for subcl in subcls:
            subcldir = os.path.join(cldir, subcl)
            rel_subcldir = os.path.join(rel_cldir, subcl)
            images = os.listdir(subcldir)
            image_dirs = [os.path.join(rel_subcldir, i) for i in images]

            for image_dir in image_dirs:
                class_files.append((cl, image_dir))

    # Process by_write directory
    write_dir = os.path.join(parent_path, 'raw_data', 'by_write')
    rel_write_dir = os.path.join('raw_data', 'by_write')
    write_parts = os.listdir(write_dir)

    for write_part in write_parts:
        writers_dir = os.path.join(write_dir, write_part)
        rel_writers_dir = os.path.join(rel_write_dir, write_part)
        writers = os.listdir(writers_dir)

        for writer in writers:
            writer_dir = os.path.join(writers_dir, writer)
            rel_writer_dir = os.path.join(rel_writers_dir, writer)
            wtypes = os.listdir(writer_dir)

            for wtype in wtypes:
                type_dir = os.path.join(writer_dir, wtype)
                rel_type_dir = os.path.join(rel_writer_dir, wtype)
                images = os.listdir(type_dir)
                image_dirs = [os.path.join(rel_type_dir, i) for i in images]

                for image_dir in image_dirs:
                    write_files.append((writer, image_dir))

    return class_files, write_files

def get_hashes(parent_path, class_file_dirs, write_file_dirs):
    """Generate hashes for all image files"""
    class_file_hashes = []
    write_file_hashes = []

    for tup in class_file_dirs:
        cclass, cfile = tup
        file_path = os.path.join(parent_path, cfile)
        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        class_file_hashes.append((cclass, cfile, chash))

    for tup in write_file_dirs:
        cclass, cfile = tup
        file_path = os.path.join(parent_path, cfile)
        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        write_file_hashes.append((cclass, cfile, chash))

    return class_file_hashes, write_file_hashes

def match_hashes(class_file_hashes, write_file_hashes):
    """Match hashes to link writers with classes"""
    class_hash_dict = {}
    for c, f, h in reversed(class_file_hashes):
        class_hash_dict[h] = (c, f)

    write_classes = []
    for w, f, h in write_file_hashes:
        write_classes.append((w, f, class_hash_dict[h][0]))

    return write_classes

def group_by_writer(write_classes):
    """Group images by writer"""
    writers = []  # each entry is a (writer, [list of (file, class)]) tuple
    cimages = []
    cw, _, _ = write_classes[0]
    
    for w, f, c in write_classes:
        if w != cw:
            writers.append((cw, cimages))
            cw = w
            cimages = [(f, c)]
        cimages.append((f, c))
    writers.append((cw, cimages))
    
    return writers

def relabel_class(c):
    """Relabel class values"""
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90:  # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61)

def data_to_json(writers, parent_path, max_writers_per_file=100):
    """Convert data to JSON format"""
    from PIL import Image
    
    # Handle different Pillow versions
    try:
        # For newer Pillow versions (10.0.0+)
        RESIZE_FILTER = Image.Resampling.LANCZOS
    except AttributeError:
        try:
            # For older Pillow versions
            RESIZE_FILTER = Image.ANTIALIAS
        except AttributeError:
            # Fallback option
            RESIZE_FILTER = Image.LANCZOS
    
    num_json = (len(writers) + max_writers_per_file - 1) // max_writers_per_file
    writer_count = 0
    json_index = 0
    
    all_data = {'users': [], 'num_samples': [], 'user_data': {}}
    
    for w, l in writers:
        all_data['users'].append(w)
        all_data['num_samples'].append(len(l))
        all_data['user_data'][w] = {'x': [], 'y': []}

        size = (28, 28)  # resize images to 28x28
        for f, c in l:
            file_path = os.path.join(parent_path, f)
            img = Image.open(file_path)
            gray = img.convert('L')
            # Use the determined resize filter
            gray.thumbnail(size, RESIZE_FILTER)
            arr = np.asarray(gray).copy()
            vec = arr.flatten()
            vec = vec / 255  # normalize pixel values
            vec = vec.tolist()

            nc = relabel_class(c)

            all_data['user_data'][w]['x'].append(vec)
            all_data['user_data'][w]['y'].append(nc)

        writer_count += 1
        
        if writer_count == max_writers_per_file or writer_count == len(writers):
            file_name = f'all_data_{json_index}.json'
            file_path = os.path.join(parent_path, 'all_data', file_name)
            
            print(f'Writing {file_name}')
            with open(file_path, 'w') as outfile:
                json.dump(all_data, outfile)

            writer_count = 0
            json_index += 1
            all_data = {'users': [], 'num_samples': [], 'user_data': {}}

def download_femnist(data_dir: str):
    """Download and preprocess FEMNIST dataset"""
    raw_data_dir = os.path.join(data_dir, "raw_data")
    intermediate_dir = os.path.join(data_dir, "intermediate")
    json_dir = os.path.join(data_dir, "all_data")
    
    dataset_url = {
        "by_class": "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
        "by_write": "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    }
    
    # Create directories
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # Download and extract datasets
    for key, url in dataset_url.items():
        zip_path = os.path.join(raw_data_dir, f"{key}.zip")
        
        if not os.path.exists(zip_path):
            print(f"Downloading {key} dataset...")
            urlretrieve(url, zip_path)
            
        print(f"Extracting {key} dataset...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
            
        os.remove(zip_path)
    
    print("Dataset downloaded and extracted. Starting preprocessing...")
    
    # Get file directories
    print("Getting file directories...")
    class_files, write_files = get_file_dirs(data_dir)
    save_obj(class_files, os.path.join(intermediate_dir, 'class_file_dirs'))
    save_obj(write_files, os.path.join(intermediate_dir, 'write_file_dirs'))
    
    # Generate hashes
    print("Generating file hashes...")
    class_file_hashes, write_file_hashes = get_hashes(data_dir, class_files, write_files)
    save_obj(class_file_hashes, os.path.join(intermediate_dir, 'class_file_hashes'))
    save_obj(write_file_hashes, os.path.join(intermediate_dir, 'write_file_hashes'))
    
    # Match hashes
    print("Matching hashes...")
    write_classes = match_hashes(class_file_hashes, write_file_hashes)
    save_obj(write_classes, os.path.join(intermediate_dir, 'write_with_class'))
    
    # Group by writer
    print("Grouping by writer...")
    writers = group_by_writer(write_classes)
    save_obj(writers, os.path.join(intermediate_dir, 'images_by_writer'))
    
    # Convert to JSON
    print("Converting to JSON format...")
    data_to_json(writers, data_dir)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download and preprocess FEMNIST dataset')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Base directory for dataset storage')
    args = parser.parse_args()
    
    download_femnist(args.data_dir)