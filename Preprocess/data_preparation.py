import zipfile
from keras.preprocessing.image import ImageDataGenerator


def load_dataset():
    path = '../Data/fer2013.zip'
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall('../Data/')


def prepare_dataset(batch_size, img_size):
    folder_path = '../Data/fer2013/'

    datagen_train = ImageDataGenerator(horizontal_flip=True,
                                       brightness_range=(0.7, 1.3),
                                       rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.1,
                                       validation_split=0.2)

    datagen_validation = ImageDataGenerator(rescale=1. / 255,
                                            validation_split=0.2)

    train_set = datagen_train.flow_from_directory(folder_path + "train",
                                                  seed=42,
                                                  target_size=(img_size, img_size),
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode='sparse',
                                                  shuffle=True,
                                                  subset='training')

    validation_set = datagen_validation.flow_from_directory(folder_path + "train",
                                                            seed=42,
                                                            target_size=(img_size, img_size),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='sparse',
                                                            shuffle=True,
                                                            subset='validation')

    datagen_test = ImageDataGenerator(rescale=1. / 255)

    test_set = datagen_test.flow_from_directory(folder_path + "test",
                                                target_size=(img_size, img_size),
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='sparse',
                                                shuffle=True)
    return train_set, validation_set, test_set
