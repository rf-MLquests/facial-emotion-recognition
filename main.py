from Preprocess.data_preparation import load_dataset, prepare_cnn_dataset, prepare_efficientnet_dataset
from Inference.classify import predict_test_images
from Training.train_cnn import build_cnn
from Training.fine_tune_efficientnets import build_efficientnet
import numpy as np
import tensorflow as tf
import yaml


def read_settings(path):
    with open(path, 'r') as file:
        args = yaml.safe_load(file)
    return args


def evaluate_cnn(test_set, num_test_samples, num_classes, image_size, channels, learning_rate):
    cnn = build_cnn((image_size, image_size, channels), num_classes, learning_rate)
    cnn.load_weights("Models/cnn/cnn.ckpt")
    labels, predictions = predict_test_images(test_set, num_test_samples, cnn)
    print("overall accuracy: ")
    print(np.where(labels == predictions)[0].shape[0] / labels.shape[0])


def evaluate_efficientnet(test_set, num_test_samples, num_classes, image_size, channels, learning_rate, use_v2=True):
    efficientnet = build_efficientnet((image_size, image_size, channels), num_classes, learning_rate, use_v2)
    if use_v2:
        efficientnet.load_weights("Models/efficientnetv2/efficientnet.ckpt")
    else:
        efficientnet.load_weights("Models/efficientnet/efficientnet.ckpt")
    labels, predictions = predict_test_images(test_set, num_test_samples, efficientnet)
    print("overall accuracy: ")
    print(np.where(labels == predictions)[0].shape[0] / labels.shape[0])


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    settings = read_settings('settings.yaml')
    # load_dataset('Data/', 'Facial_emotion_images.zip')
    # train, val, test = prepare_cnn_dataset('Data/Facial_emotion_images/',
    #                                        settings['cnn']['batch_size'],
    #                                        settings['cnn']['image_size'])
    # evaluate_cnn(test, 10,
    #              settings['cnn']['num_classes'],
    #              settings['cnn']['image_size'],
    #              settings['cnn']['channels'],
    #              settings['cnn']['learning_rate'])
    train, val, test = prepare_efficientnet_dataset('Data/Facial_emotion_images/',
                                                    settings['efficientnet']['batch_size'],
                                                    settings['efficientnet']['image_size'])
    evaluate_efficientnet(test, 10,
                          settings['efficientnet']['num_classes'],
                          settings['efficientnet']['image_size'],
                          settings['efficientnet']['channels'],
                          settings['efficientnet']['learning_rate'])
