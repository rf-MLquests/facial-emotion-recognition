from Inference.classify import select_test_images, predict_test_images
from Training.train_cnn import build_cnn
import numpy as np
import tensorflow as tf


def main():
    test_images, labels = select_test_images(5, "Data/Facial_emotion_images/")
    model = build_cnn()
    model.load_weights("Models/cnn/cnn.ckpt")
    predictions = predict_test_images(test_images, model)
    print(labels)
    print(predictions)
    print("overall accuracy: ")
    print(np.where(labels == predictions)[0].shape[0] / labels.shape[0])


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()
