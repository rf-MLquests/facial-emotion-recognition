from Preprocess.data_preparation import load_dataset, prepare_cnn_dataset
from Inference.classify import predict_test_images
from Training.train_cnn import build_cnn
import numpy as np
import tensorflow as tf


def evaluate_cnn(test_set, num):
    cnn = build_cnn((48, 48, 1), 7, 0.001, 0.0001)
    cnn.load_weights("Models/cnn/cnn.ckpt")
    labels, predictions = predict_test_images(test_set, num, cnn)
    print("overall accuracy: ")
    print(np.where(labels == predictions)[0].shape[0] / labels.shape[0])


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    # load_dataset('Data/', 'Facial_emotion_images.zip')
    train, val, test = prepare_cnn_dataset('Data/', 128, 48)
    evaluate_cnn(test, 10)
