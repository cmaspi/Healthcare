import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from tqdm import tqdm
from model import get_model
from preprocessing import *


parent_dir = './Data/In-lab/'
paths = sorted(list(os.listdir(parent_dir)))


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def calculate_metrics(valX, valY, model_path):
    model = tf.keras.models.load_model(f'./chkpts/{model_path}')
    pred = []
    results = {}
    for i in range(len(valX)):
        dX, dY = valX[i].reshape((1,) + valX[i].shape + (1,)), valY[i:i + 1]
        p = model.predict(dX)[0, 0]
        pred.append(p)

    pred = np.round(pred)
    results['precision'] = precision_score(valY, pred)
    results['recall'] = recall_score(valY, pred)
    results['f1'] = f1_score(valY, pred)

    return results


def normalize_data(data, mode):
    """inplace operation"""

    def minimum(data):
        return min(min(data, key=lambda x: min(x)))

    def maximum(data):
        return max(max(data, key=lambda x: max(x)))

    if mode == '0-1':
        mini = minimum(data)
        for subarr in data:
            subarr -= mini
        maxi = maximum(data)
        for subarr in data:
            subarr /= maxi
    if mode == 'z':
        raise NotImplementedError("not implemented yet")


def return_dataset(paths):
    data = []
    activities_list = []
    labels_list = []
    ema_list = []
    for path in tqdm(paths):
        dataX, ema, labels, activities = get_data_activity_chunks(parent_dir + path, sampling=3)
        normalize_data(dataX, mode='0-1')
        data.extend(dataX)
        ema_list.append(ema)
        labels_list.append(labels)
        activities_list.append(activities)

    return data, np.concatenate(ema_list, axis=0), np.concatenate(labels_list, axis=0), np.concatenate(activities_list,
                                                                                                       axis=0)


for val_idx in range(len(paths)):
    training_paths = paths[:val_idx] + paths[val_idx + 1:]
    validation_paths = paths[val_idx: val_idx + 1]

    print('#' * 20 + '\n' + f'{val_idx}\n' + '#' * 20)

    trainX, trainEMA, trainY, trainActivities = return_dataset(training_paths)
    valX, valEMA, valY, valActivities = return_dataset(validation_paths)
    print(f'fraction of stressful activities: {trainY.sum()/trainY.size}')
    model = get_model(input_size=None)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'],
                  )
    weights = class_weight.compute_class_weight('balanced',
                                                classes=np.unique(trainY),
                                                y=trainY)
    weights = {
        0: weights[0],
        1: weights[1]
    }
    num_epochs = 100
    max_vacc = -1
    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch: {epoch + 1}')
        overall_loss, overall_acc = [], []
        for i in range(len(trainX)):
            dX, dY = trainX[i].reshape((1,) + trainX[i].shape + (1,)), trainY[i:i + 1]
            loss, acc = model.train_on_batch(dX, dY, class_weight=weights)
            overall_acc.append(acc)
            overall_loss.append(loss)
        print(f'accuracy: {np.mean(overall_acc)}, loss: {np.mean(overall_loss)}')

        val_overall_loss, val_overall_acc = [], []
        for i in range(len(valX)):
            dX, dY = valX[i].reshape((1,) + valX[i].shape + (1,)), valY[i:i + 1]
            loss, acc = model.test_on_batch(dX, dY)
            val_overall_acc.append(acc)
            val_overall_loss.append(loss)
        print(f'val accuracy: {np.mean(val_overall_acc)}, val loss: {np.mean(val_overall_loss)}')
        val_acc = np.mean(val_overall_acc)
        if val_acc > max_vacc:
            max_vacc = val_acc
            model.save(f"./chkpts/LOSO/{model.name}_{validation_paths[0]}")

    tf.keras.backend.clear_session()
    del model

    # saving metrics
    results = calculate_metrics(valX, valY, validation_paths[0])
    with open('./Results/loso_dl.csv', 'a') as f:
        print(f'{validation_paths[0]}, {results["precision"]}, {results["recall"]}, {results["f1"]}', file=f)
