import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset_randomly_into_train_test(source_dir, train_dir, test_dir, train_size=0.7):
    # Create train and test directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get the list of class subdirectories
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Shuffle images randomly
        train_images, test_images = train_test_split(images, train_size=train_size, random_state=42)
        
        # Create class subdirectories in train and test directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Copy the images to the respective directories
        for img in train_images:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))
        
        for img in test_images:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(test_dir, class_name, img))
        
        print(f"Class '{class_name}': {len(train_images)} images copied to train, {len(test_images)} images copied to test.")

def split_dataset_alphabetically_into_train_test(source_dir, train_dir, test_dir, train_size=0.7):
    # Create train and test directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get the list of class subdirectories
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Sort images alphabetically
        images.sort()
        
        # Check if there are any images to split
        if len(images) == 0:
            print(f"No images found in class '{class_name}', skipping...")
            continue
        
        # Calculate split index
        split_idx = int(len(images) * train_size)
        
        # Split the images into train and test sets
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class subdirectories in train and test directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Copy the images to the respective directories
        for img in train_images:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))
        
        for img in test_images:
            shutil.copy2(os.path.join(class_dir, img), os.path.join(test_dir, class_name, img))
        
        print(f"Class '{class_name}': {len(train_images)} images copied to train, {len(test_images)} images copied to test.")


if __name__ == "__main__":
    source_dir  = "./data/train4/"            # Path to the source directory containing class subdirectories
    train_dir   = "./data/train4-random/train/"      # Path to the train directory
    test_dir    = "./data/train4-random/test/"       # Path to the test directory
    train_size  = 0.8                                # Proportion of images to use for training

    # split_dataset_randomly_into_train_test(source_dir, train_dir, test_dir, train_size)

    split_dataset_alphabetically_into_train_test(source_dir, train_dir, test_dir, train_size)
