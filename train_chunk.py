import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import class_weight
from tqdm import tqdm
from model import get_model
from preprocessing import *
from sklearn.model_selection import train_test_split

parent_dir = "./Data/In-lab/"
paths = sorted(list(os.listdir(parent_dir)))


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if type(a) is list:
        return [a[i] for i in p], b[p]
    return a[p], b[p]


def calculate_metrics(valX, valY, model_path):
    model = tf.keras.models.load_model(f"./chkpts/LOSO/{model_path}")
    pred = []
    results = {}
    for i in range(len(valX)):
        dX, dY = valX[i].reshape((1, ) + valX[i].shape + (1, )), valY[i:i + 1]
        p = model.predict(dX)[0, 0]
        pred.append(p)

    pred = np.round(pred)
    results["precision"] = precision_score(valY, pred)
    results["recall"] = recall_score(valY, pred)
    results["f1"] = f1_score(valY, pred)

    return results


def normalize_data(data, mode):
    """inplace operation"""

    def minimum(data):
        return min(min(data, key=lambda x: min(x)))

    def maximum(data):
        return max(max(data, key=lambda x: max(x)))

    if mode == "0-1":
        mini = minimum(data)
        for subarr in data:
            subarr -= mini
        maxi = maximum(data)
        for subarr in data:
            subarr /= maxi
    if mode == "z":
        raise NotImplementedError("not implemented yet")


def return_dataset(paths):
    data = []
    activities_list = []
    labels_list = []
    ema_list = []
    for path in tqdm(paths):
        dataX, ema, labels, activities = get_data_activity_chunks(parent_dir +
                                                                  path,
                                                                  sampling=15)
        normalize_data(dataX, mode="0-1")
        data.extend(dataX)
        ema_list.append(ema)
        labels_list.append(labels)
        activities_list.append(activities)

    return (
        data,
        np.concatenate(labels_list, axis=0),
    )


def train_iteration(model, trainX, trainY, valX, valY, weights):
    overall_loss, overall_acc = [], []
    for i in range(len(trainX)):
        dX, dY = trainX[i].reshape((1, -1, 1)), trainY[i:i + 1]
        loss, acc = model.train_on_batch(dX, dY, class_weight=weights)
        overall_acc.append(acc)
        overall_loss.append(loss)
    print(f"accuracy: {np.mean(overall_acc)}, loss: {np.mean(overall_loss)}")

    predicted = []
    for i in range(len(valX)):
        dX, dY = valX[i].reshape((1, ) + valX[i].shape + (1, )), valY[i:i + 1]
        pred = np.round(model.predict(dX, verbose=0)[0, 0])
        predicted.append(pred)
    val_acc = accuracy_score(valY, predicted)
    val_precision = precision_score(valY, predicted)
    val_recall = recall_score(valY, predicted)
    val_f1 = f1_score(valY, predicted)
    return val_acc, val_precision, val_recall, val_f1


def train():
    trainX, trainY = return_dataset(paths)
    trainX, trainY = unison_shuffled_copies(trainX, trainY)
    trainX, valX, trainY, valY = train_test_split(trainX, trainY)

    print(f"fraction of stressful activities: {trainY.sum()/trainY.size}")
    model = get_model(input_size=None)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    weights = class_weight.compute_class_weight("balanced",
                                                classes=np.unique(trainY),
                                                y=trainY)
    weights = {0: weights[0], 1: weights[1]}
    num_epochs = 100
    max_vf1 = -1
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch + 1}")

        val_acc, val_precision, val_recall, val_f1 = train_iteration(
            model, trainX, trainY, valX, valY, weights)

        print(
            f"val accuracy: {val_acc}\nval precision: {val_precision}\nval recall: {val_recall}\nval f1: {val_f1}"
        )

        if val_f1 > max_vf1:
            max_vf1 = val_f1
            model.save_weights(f"./chkpts/{model.name}")

    # Evaluating the final thingy
    model.load_weights(f"./chkpts/{model.name}")
    predicted = []
    for i in range(len(valX)):
        dX = valX[i].reshape((1, -1, 1))
        pred = np.round(model.predict(dX, verbose=0)[0, 0])
        predicted.append(pred)
    val_acc = accuracy_score(valY, predicted)
    val_precision = precision_score(valY, predicted)
    val_recall = recall_score(valY, predicted)
    val_f1 = f1_score(valY, predicted)
    print(
        f"val accuracy: {val_acc}\nval precision: {val_precision}\nval recall: {val_recall}\nval f1: {val_f1}"
    )


if __name__ == '__main__':
    train()
