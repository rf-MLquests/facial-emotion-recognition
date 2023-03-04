import zipfile
import matplotlib.pyplot as plt
import os

from keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator


def load_dataset():
    path = '../Data/Facial_emotion_images.zip'
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall('../Data/')


def demo_images():
    folder_path = '../Data/Facial_emotion_images/'
    image_size = 48
    expressions = ['happy', 'sad', 'neutral', 'surprise']
    plt.figure(figsize=(8, 8))
    for expression in expressions:
        for i in range(1, 10, 1):
            plt.subplot(3, 3, i)

            img = load_img(folder_path + "train/" + expression + "/" +
                           os.listdir(folder_path + "train/" + expression)[i], target_size=(image_size, image_size))
            plt.imshow(img)
        plt.show()


def prepare_dataset():
    folder_path = '../Data/Facial_emotion_images/'
    batch_size = 32
    img_size = 48

    datagen_train = ImageDataGenerator(horizontal_flip=True,
                                       brightness_range=(0., 2.),
                                       rescale=1. / 255,
                                       shear_range=0.3)

    train_set = datagen_train.flow_from_directory(folder_path + "train",
                                                  target_size=(img_size, img_size),
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

    datagen_validation = ImageDataGenerator(horizontal_flip=True,
                                            brightness_range=(0., 2.),
                                            rescale=1. / 255,
                                            shear_range=0.3)

    validation_set = datagen_validation.flow_from_directory(folder_path + "validation",
                                                            target_size=(img_size, img_size),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=True)

    datagen_test = ImageDataGenerator(horizontal_flip=True,
                                      brightness_range=(0., 2.),
                                      rescale=1. / 255,
                                      shear_range=0.3)

    test_set = datagen_test.flow_from_directory(folder_path + "test",
                                                target_size=(img_size, img_size),
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=True)
    return train_set, validation_set, test_set
