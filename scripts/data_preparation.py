import os
import tensorflow as tf

def parse_test_file(test_file_path, root_dir):
    image_paths = []
    labels = []
    label_mapping = {0: "glass", 1: "paper", 2: "cardboard", 3: "plastic", 4: "metal", 5: "trash"} 

    with open(test_file_path, "r") as f:
        for line in f:
            image_name, label = line.strip().split()
            label = int(label) - 1  # 1-indexed to 0-indexed
            folder_name = label_mapping[label]
            image_path = os.path.join(root_dir, "Garbage_classification", "garbages", folder_name, image_name)

            if os.path.exists(image_path):  # only if image exists
                image_paths.append(image_path)
                labels.append(label)

    return image_paths, labels

def load_images(image_paths, labels, img_size=(150, 150)):
    images = []
    for img_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # 正規化
        images.append(img)
    
    # TensorFlowデータセット形式に変換
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    return images, labels
