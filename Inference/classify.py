import os
from keras.utils import load_img, img_to_array
import numpy as np


def select_test_images(num_per_class, folder_path):
    label_map = {'angry': 0,
                 'disgust': 1,
                 'fear': 2,
                 'happy': 3,
                 'neutral': 4,
                 'sad': 5,
                 'surprise': 6}
    img_size = 48
    images = []
    labels = []
    expressions = ['angry',
                   'disgust',
                   'fear',
                   'happy',
                   'neutral',
                   'sad',
                   'surprise']
    for expression in expressions:
        for i in range(1, num_per_class + 1, 1):
            img = load_img(folder_path + "test/" + expression + "/" + os.listdir(folder_path + "test/" + expression)[i],
                           target_size=(img_size, img_size))
            images.append(img)
            labels.append(label_map[expression])
    return images, np.array(labels)


def predict_test_images(images, model):
    input_batch = []
    for img in images:
        img_array = img_to_array(img)
        input_batch.append(img_array)
    input_batch = np.array(input_batch)
    output = model.predict(input_batch)
    predictions = np.argmax(output, axis=1)
    return predictions
