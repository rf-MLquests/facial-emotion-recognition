def predict_test_images(test_set, model):
    loss_and_accuracy = model.evaluate(test_set)
    return loss_and_accuracy[1]
