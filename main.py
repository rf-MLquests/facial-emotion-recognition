from Preprocess.data_preparation import load_dataset, prepare_cnn_dataset, prepare_efficientnet_dataset, \
    prepare_vit_dataloaders
from Inference.classify import predict_test_images
from Training.train_cnn import build_cnn
from Training.fine_tune_efficientnets import build_efficientnet
from Training.fine_tune_vit import initialize_model
import numpy as np
import tensorflow as tf
import torch
import yaml


def read_settings(path):
    with open(path, 'r') as file:
        args = yaml.safe_load(file)
    return args


def evaluate_cnn(test_set, num_classes, image_size, channels, learning_rate):
    cnn = build_cnn((image_size, image_size, channels), num_classes, learning_rate)
    cnn.load_weights("Models/cnn/cnn.ckpt")
    accuracy = predict_test_images(test_set, cnn)
    print("overall accuracy: ")
    print(accuracy)


def evaluate_efficientnet(test_set, num_classes, image_size, channels, learning_rate, use_v2=True):
    efficientnet = build_efficientnet((image_size, image_size, channels), num_classes, learning_rate, use_v2)
    if use_v2:
        efficientnet.load_weights("Models/efficientnetv2/efficientnet.ckpt")
    else:
        efficientnet.load_weights("Models/efficientnet/efficientnet.ckpt")
    accuracy = predict_test_images(test_set, efficientnet)
    print("overall accuracy: ")
    print(accuracy)


def evaluate_vit(test_loader, num_classes):
    model, input_size = initialize_model(num_classes, True)
    checkpoint_path = "Models/vit/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path + "epoch-16.pth", map_location=device))
    model.eval()
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        pred = outputs.cpu().detach().numpy()
        y_true = labels.cpu().detach().numpy()
    predictions = np.argmax(pred, axis=1)
    correct_indices = np.where(y_true == predictions)[0]
    accuracy = correct_indices.shape[0] / test_loader.batch_size
    print("overall accuracy: ")
    print(accuracy)


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    settings = read_settings('settings.yaml')
    # load_dataset('Data/', 'Facial_emotion_images.zip')
    # train, val, test = prepare_efficientnet_dataset('Data/Facial_emotion_images/',
    #                                                 settings['efficientnet']['batch_size'],
    #                                                 settings['efficientnet']['image_size'])
    # evaluate_efficientnet(test,
    #                       settings['efficientnet']['num_classes'],
    #                       settings['efficientnet']['image_size'],
    #                       settings['efficientnet']['channels'],
    #                       settings['efficientnet']['learning_rate'])

    train, val, test = prepare_vit_dataloaders('Data/Facial_emotion_images/',
                                               settings['vit']['image_size'],
                                               settings['vit']['batch_size'])
    evaluate_vit(test, settings['vit']['num_classes'])
