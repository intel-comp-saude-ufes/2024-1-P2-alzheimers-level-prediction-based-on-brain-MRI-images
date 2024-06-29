import os
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
import shutil

def get_mri_augmentation_sequence():
    """
    Retorna a sequência de augmentations adequada para imagens de ressonância magnética cerebral.

    Returns:
        iaa.Sequential: Sequência de augmentations.
    """
    # Definir a sequência de augmentações a serem aplicadas
    seq = iaa.Sequential([
        iaa.Fliplr(0.1),                                    # Flip horizontal com probabilidade de 10%
        iaa.Affine(
            rotate=(-10, 10),                               # Rotação aleatória entre -10 e 10 graus
            shear=(-5, 5),                                  # Inclinação aleatória entre -5 e 5 graus
            scale=(0.9, 1.1)                                # Escalamento aleatório entre 90% e 110%
        ),
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),     # Adicionar ruído gaussiano leve
        iaa.Multiply((0.9, 1.1)),                           # Alterar o brilho
        iaa.LinearContrast((0.9, 1.1)),                     # Alterar o contraste
    ])
    return seq

def augment_image(image):
    """
    Realiza data augmentation em uma imagem de ressonância magnética cerebral.

    Args:
        image (np.array): Imagem a ser aumentada.

    Returns:
        augmented_image (np.array): Imagem aumentada.
    """

    seq = get_mri_augmentation_sequence()

    augmented_image = seq(image=image)

    return augmented_image

def augment_and_save_images(input_dir, output_dir, num_augmented_images=10):
    """
    Aplica augmentação em imagens de ressonância magnética e salva as imagens aumentadas
    na estrutura de diretórios correspondente.

    Args:
        input_dir (str): Diretório contendo as imagens originais.
        output_dir (str): Diretório onde as imagens aumentadas serão salvas.
        num_augmented_images (int): Número de imagens aumentadas a serem geradas para cada imagem original.
    """
    # Criar a sequência de augmentações
    augmentation_sequence = get_mri_augmentation_sequence()

    # Criar a estrutura de diretórios de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Percorrer cada subdiretório na pasta de entrada
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            output_category_path = os.path.join(output_dir, category)
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            # Processar cada imagem no subdiretório
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                image = Image.open(img_path).convert('L')  # Converter para escala de cinza
                image_np = np.array(image)

                # Gerar imagens aumentadas
                for i in range(num_augmented_images):
                    augmented_image_np = augmentation_sequence(image=image_np)
                    augmented_image = Image.fromarray(augmented_image_np)

                    # Salvar a imagem aumentada
                    base_name, ext = os.path.splitext(img_name)
                    augmented_img_name = f"{base_name}_aug_{i + 1}{ext}"
                    augmented_img_path = os.path.join(output_category_path, augmented_img_name)
                    augmented_image.save(augmented_img_path)

def balance_classes_with_augmentation(input_dir, output_dir):
    """
    Gera augmentações para equilibrar o número de imagens em cada classe.

    Args:
        input_dir (str): Diretório contendo as imagens originais.
        output_dir (str): Diretório onde as imagens aumentadas serão salvas.
    """
    # Contar o número de imagens em cada classe
    class_counts = {}
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            class_counts[category] = len(os.listdir(category_path))

    # Determinar o número máximo de imagens em qualquer classe
    max_count = max(class_counts.values())

    # Aplicar augmentação para balancear as classes
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
                image = Image.open(img_path).convert('L')  # Converter para escala de cinza
                image_np = np.array(image)

                # Gerar augmentações para cada imagem existente
                for i in range(images_per_existing_image):
                    augmented_image_np = get_mri_augmentation_sequence()(image=image_np)
                    augmented_image = Image.fromarray(augmented_image_np)
                    base_name, ext = os.path.splitext(img_name)
                    augmented_img_name = f"{base_name}_aug_{i + 1}{ext}"
                    augmented_img_path = os.path.join(output_category_path, augmented_img_name)
                    augmented_image.save(augmented_img_path)

            # Se houver imagens adicionais necessárias, gere-as a partir das primeiras imagens existentes
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
    # Caminhos de entrada e saída
    input_directory = "data/train"
    output_directory = "data_augmented"

    # Aplicar augmentação e salvar as imagens
    augment_and_save_images(input_directory, output_directory, 10)
