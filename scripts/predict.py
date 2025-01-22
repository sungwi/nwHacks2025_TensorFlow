from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model_path, img_path):
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    print("Prediction:", prediction)
    predicted_class = np.argmax(prediction)

    class_mapping = {0: "glass", 1: "paper", 2: "cardboard", 3: "plastic", 4: "metal", 5: "trash"} 
    return class_mapping[predicted_class]

# 使用例
result = predict_image("waste_classifier_model.h5", "garbage_classification/Garbage_classification/garbages/cardboard/cardboard3.jpg")
print("Predicted class:", result)


# # テストデータでモデルを評価
# model = load_model("waste_classifier_model.h5")
# test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
