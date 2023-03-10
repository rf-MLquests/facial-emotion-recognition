import numpy as np


def predict_test_images(test_set, num_samples, model):
    batch, labels = test_set.next()
    input_batch = batch[:num_samples]
    labels = labels[:num_samples]
    predictions = np.argmax(model.predict(input_batch), axis=1)
    print("Labels for sample images: ")
    print(labels)
    print("Model predictions for sample images: ")
    print(predictions)
    return labels, predictions
