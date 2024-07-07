import os
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
import shutil

def get_mri_augmentation_sequence():
    """
    Returns the sequence of augmentations suitable for brain MRI images

    INPUT:
        None

    OUTPUT:
        iaa.Sequential: Augmentation sequence
    """

    # Define the sequence of augmentations to be applied
    seq = iaa.Sequential([
        iaa.Fliplr(0.1),                                # Horizontal flip with 10% probability
        iaa.Affine(
            rotate=(-10, 10),                           # Random rotation between -10 and 10 degrees
            shear=(-5, 5),                              # Random tilt between -5 and 5 degrees
            scale=(0.9, 1.1)                            # Random scaling between 90% and 110%
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)), # Add Light Gaussian Noise
        iaa.Multiply((0.9, 1.1)),                       # Change the brightness
        iaa.LinearContrast((0.9, 1.1)),                 # Change the contrast
    ])
    return seq


def augment_image(image):
    """
    Performs data augmentation on a brain magnetic resonance image

    INPUT:
        image (np.array): Image to be augmented

    OUTPUT:
        augmented_image (np.array): Augmented image
    """

    seq = get_mri_augmentation_sequence()
    augmented_image = seq(image=image)
    return augmented_image


def augment_and_save_images(input_dir, output_dir, num_augmented_images=10):
    """
    Applies augmentation to MRI images and saves the augmented images in the corresponding directory structure

    INPUT:
        input_dir (str): Directory containing the original images
        output_dir (str): Directory where the augmented images will be saved
        num_augmented_images (int): Number of augmented images to generate for each original image

    OUTPUT:
        None
    """

    # Create the augmentation sequence
    augmentation_sequence = get_mri_augmentation_sequence()

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cycle through each subdirectory in the input folder
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            output_category_path = os.path.join(output_dir, category)
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            # Process each image in the category
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                image_np = np.array(image)

                # Generate augmented images
                for i in range(num_augmented_images):
                    augmented_image_np = augmentation_sequence(image=image_np)
                    augmented_image = Image.fromarray(augmented_image_np)

                    # Save the augmented image
                    base_name, ext = os.path.splitext(img_name)
                    augmented_img_name = f"{base_name}_aug_{i + 1}{ext}"
                    augmented_img_path = os.path.join(output_category_path, augmented_img_name)
                    augmented_image.save(augmented_img_path)


def balance_classes_with_augmentation(input_dir, output_dir):
    """
    Generates augmentations to balance the number of images in each class

    INPUT:
        input_dir (str): Directory containing the original images
        output_dir (str): Directory where the enlarged images will be saved

    OUTPUT:
        None
    """

    # Count the number of images in each class
    class_counts = {}
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            class_counts[category] = len(os.listdir(category_path))

    # Determine the maximum number of images in any class
    max_count = max(class_counts.values())

    # Apply augmentation to balance classes
    for category in class_counts:
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)
        
        existing_images = os.listdir(category_path)
        num_existing_images = len(existing_images)
        num_images_to_generate = max_count - num_existing_images

        if num_images_to_generate > 0:
            images_per_existing_image = num_images_to_generate // num_existing_images
            additional_images_needed = num_images_to_generate % num_existing_images

            for img_name in existing_images:
                img_path = os.path.join(category_path, img_name)
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                image_np = np.array(image)

                # Generate augmentations for each existing image
                for i in range(images_per_existing_image):
                    augmented_image_np = get_mri_augmentation_sequence()(image=image_np)
                    augmented_image = Image.fromarray(augmented_image_np)
                    base_name, ext = os.path.splitext(img_name)
                    augmented_img_name = f"{base_name}_aug_{i + 1}{ext}"
                    augmented_img_path = os.path.join(output_category_path, augmented_img_name)
                    augmented_image.save(augmented_img_path)

            # If there are additional images required, generate them from the first existing images
            for i in range(additional_images_needed):
                img_name = existing_images[i]
                img_path = os.path.join(category_path, img_name)
                image = Image.open(img_path).convert('L')
                image_np = np.array(image)
                augmented_image_np = get_mri_augmentation_sequence()(image=image_np)
                augmented_image = Image.fromarray(augmented_image_np)
                base_name, ext = os.path.splitext(img_name)
                augmented_img_name = f"{base_name}_aug_additional_{i + 1}{ext}"
                augmented_img_path = os.path.join(output_category_path, augmented_img_name)
                augmented_image.save(augmented_img_path)


if __name__ == '__main__':
    # Entry and exit paths
    input_directory = "data/train"
    output_directory = "data_augmented"

    # Apply augmentation and save images
    augment_and_save_images(input_directory, output_directory, 10)