from Preprocess.data_preparation import load_dataset, prepare_dataset
from Inference.classify import predict_test_images
from Training.train_cnn import build_cnn
from Training.train_vit import build_vit
import numpy as np
import tensorflow as tf


def evaluate_cnn(test_set, num):
    cnn = build_cnn((48, 48, 3), 7, 0.001, 0.0001)
    cnn.load_weights("Models/cnn/cnn.ckpt")
    labels, predictions = predict_test_images(test_set, num, cnn)
    print("overall accuracy: ")
    print(np.where(labels == predictions)[0].shape[0] / labels.shape[0])


def evaluate_vit(test_set, num):
    vit = build_vit((48, 48, 3), 7, 6, 4, 8, 0.001, 0.0001)
    vit.load_weights("Models/cnn/cnn.ckpt")
    labels, predictions = predict_test_images(test_set, num, vit)
    print("overall accuracy: ")
    print(np.where(labels == predictions)[0].shape[0] / labels.shape[0])


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    # load_dataset('Data/', 'fer2013.zip')
    train, val, test = prepare_dataset('Data/', 128, 48)
    evaluate_cnn(test, 10)
    # evaluate_vit(test, 10)
