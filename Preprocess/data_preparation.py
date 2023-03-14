from __future__ import print_function
from __future__ import division
import zipfile
from keras.preprocessing.image import ImageDataGenerator
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, RandomRotation, ToTensor
from torch.utils.data import DataLoader


def load_dataset(path, filename):
    with zipfile.ZipFile(path + filename, 'r') as zip_ref:
        zip_ref.extractall(path)


def prepare_cnn_dataset(folder_path, image_size, batch_size):
    datagen_train = ImageDataGenerator(horizontal_flip=True,
                                       brightness_range=(0.7, 1.3),
                                       rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.1)

    train_set = datagen_train.flow_from_directory(folder_path + "train",
                                                  target_size=(image_size, image_size),
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

    datagen_validation = ImageDataGenerator(rescale=1. / 255)

    validation_set = datagen_validation.flow_from_directory(folder_path + "validation",
                                                            target_size=(image_size, image_size),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=True)

    datagen_test = ImageDataGenerator(rescale=1. / 255)

    test_set = datagen_test.flow_from_directory(folder_path + "test",
                                                target_size=(image_size, image_size),
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=True)
    return train_set, validation_set, test_set


def prepare_efficientnet_dataset(folder_path, image_size, batch_size):
    datagen_train = ImageDataGenerator(horizontal_flip=True,
                                       brightness_range=(0.7, 1.3),
                                       shear_range=0.2,
                                       zoom_range=0.1)

    train_set = datagen_train.flow_from_directory(folder_path + "train",
                                                  target_size=(image_size, image_size),
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

    datagen_validation = ImageDataGenerator()

    validation_set = datagen_validation.flow_from_directory(folder_path + "validation",
                                                            target_size=(image_size, image_size),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=True)

    datagen_test = ImageDataGenerator()

    test_set = datagen_test.flow_from_directory(folder_path + "test",
                                                target_size=(image_size, image_size),
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=True)
    return train_set, validation_set, test_set


def prepare_vit_dataloaders(folder_path, image_size, batch_size):
    train_transform = Compose([Resize((image_size, image_size)),
                               RandomHorizontalFlip(p=0.5),
                               RandomRotation(degrees=(15, 35)),
                               ToTensor(),
                               Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_transform = Compose([Resize((image_size, image_size)),
                                    ToTensor(),
                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = Compose([Resize((image_size, image_size)),
                              ToTensor(),
                              Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_set = ImageFolder(root=folder_path + "train",
                            transform=train_transform)

    validation_set = ImageFolder(root=folder_path + "validation",
                                 transform=validation_transform)

    test_set = ImageFolder(root=folder_path + "test",
                           transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader, test_loader
