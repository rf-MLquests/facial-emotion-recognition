from Inference.classify import select_test_images, predict_test_images
import numpy as np


def main():
    test_images, labels = select_test_images(5, "Data/Facial_emotion_images/")
    predictions = predict_test_images(test_images, "Models/CNN_model.h5")
    print(labels)
    print(predictions)
    print("overall accuracy: ")
    print(np.where(labels == predictions)[0].shape[0] / labels.shape[0])


if __name__ == "__main__":
    main()
