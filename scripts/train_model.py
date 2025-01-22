import os
import tensorflow as tf
from data_preparation import parse_test_file, load_images

# データセットの準備
root_dir = "garbage_classification"

# 各ファイルから画像パスとラベルを取得
train_file_path = os.path.join(root_dir, "one-indexed-files-notrash_test.txt")
val_file_path = os.path.join(root_dir, "one-indexed-files-notrash_train.txt")
test_file_path = os.path.join(root_dir, "one-indexed-files-notrash_val.txt")

train_image_paths, train_labels = parse_test_file(train_file_path, root_dir)
val_image_paths, val_labels = parse_test_file(val_file_path, root_dir)
test_image_paths, test_labels = parse_test_file(test_file_path, root_dir)

# データセットをロード
train_images, train_labels = load_images(train_image_paths, train_labels)
val_images, val_labels = load_images(val_image_paths, val_labels)
test_images, test_labels = load_images(test_image_paths, test_labels)

# TensorFlowデータセット形式に変換
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=len(train_images)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# モデル構築
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  
])

# モデルコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# モデルトレーニング
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# モデル保存
model.save("waste_classifier_model.h5")
