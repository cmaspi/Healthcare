import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import class_weight
from tqdm import tqdm
from model import get_model_1min
from preprocessing import *
from multiprocessing import Process
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

parent_dir = "./Data/In-lab/"
paths = sorted(list(os.listdir(parent_dir)))
interval_duration = 2
vec_size = int(15_000 * interval_duration)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def return_dataset(paths):
    data = []

    for path in tqdm(paths):
        temp = get_data(parent_dir + path, interval_duration=interval_duration)
        dataX = temp[:, :vec_size].astype(float)
        remaining = temp[:, vec_size:]
        # z score normalization
        dataX -= dataX.mean()
        dataX /= np.std(dataX)
        temp = np.concatenate([dataX, remaining], axis=1)
        data.append(temp)
    data = np.concatenate(data, axis=0)
    dataX = data[:, :vec_size].astype(float)
    dataY = data[:, -1:].astype(float)
    dataX, dataY = unison_shuffled_copies(dataX, dataY)
    return dataX, dataY


def train_loso(val_idx, fine_tune=False):
    training_paths = paths[:val_idx] + paths[val_idx + 1:]
    validation_paths = paths[val_idx:val_idx + 1]
    log_file = open(f'./logs/LOSO/{validation_paths[0]}.txt', 'w')

    print("#" * 20 + "\n" + f"{val_idx}\n" + "#" * 20, file=log_file)

    trainX, trainY = return_dataset(training_paths)
    valX, valY = return_dataset(validation_paths)
    print(trainX.shape, trainY.shape)
    print(f"fraction of stressful activities: {trainY.sum()/trainY.size}",
          file=log_file)
    model = get_model_1min(input_size=vec_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.metrics.Precision(),
            tf.metrics.Recall(),
            tf.keras.metrics.F1Score(threshold=0.5)
        ],
    )
    weights = class_weight.compute_class_weight("balanced",
                                                classes=np.unique(trainY),
                                                y=trainY[:, 0])
    weights = {0: weights[0], 1: weights[1]}
    num_epochs = 100
    earlyStopping = EarlyStopping(monitor='val_loss',
                                  patience=25,
                                  verbose=0,
                                  mode='min')
    mcp_save = ModelCheckpoint(
        f'chkpts/LOSO/{model.name}_{validation_paths[0]}.keras',
        save_best_only=True,
        monitor='val_f1_score',
        mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       patience=7,
                                       verbose=1,
                                       min_delta=1e-4,
                                       mode='min')

    model.fit(trainX,
              trainY,
              validation_data=(valX, valY),
              epochs=num_epochs,
              callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
              class_weight=weights,
              batch_size=16,
              verbose=2)

    if fine_tune:
        positive_indices = np.where(valY == 1)[0]
        negative_indices = np.where(valY == 0)[0]
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        size_negative = len(negative_indices)
        size_positive = len(positive_indices)
        negative_idx_split = int(size_negative * 0.1)
        positive_idx_split = int(size_positive * 0.1)
        fine_tune_indices = np.concatenate([
            positive_indices[:positive_idx_split],
            negative_indices[:negative_idx_split]
        ],
                                           axis=0)
        val_indices = np.concatenate([
            positive_indices[positive_idx_split:],
            negative_indices[negative_idx_split:]
        ],
                                     axis=0)
        fineX, fineY = valX[fine_tune_indices], valY[fine_tune_indices]
        new_valX, new_valY = valX[val_indices], valY[val_indices]

        model.load_weights(
            f'./chkpts/LOSO/{model.name}_{validation_paths[0]}.keras')
        pred = model.predict(valX).reshape((-1, ))  # flattening
        pred = np.round(pred)
        results = {}
        results['precision'] = precision_score(valY, pred)
        results['recall'] = recall_score(valY, pred)
        results['f1'] = f1_score(valY, pred)
        print(results)
        
        model.fit(fineX,
                  fineY,
                  validation_data=(new_valX, new_valY),
                  epochs=10,
                  verbose=2,
                  class_weight=weights)

    # saving metrics
    pred = model.predict(new_valX).reshape((-1, ))  # flattening
    pred = np.round(pred)
    results = {}
    results['precision'] = precision_score(new_valY, pred)
    results['recall'] = recall_score(new_valY, pred)
    results['f1'] = f1_score(new_valY, pred)

    with open("./Results/loso_dl_corrected_intervals.csv", "a") as f:
        print(
            f'{validation_paths[0]}, {results["precision"]}, {results["recall"]}, {results["f1"]}',
            file=f,
        )
    log_file.close()
    tf.keras.backend.clear_session()
    del model


processes = []
num_parallel_processes = 2
for val_idx in range(len(paths)):
    if len(processes) == num_parallel_processes:
        for p in processes:
            p.join()
        processes = []
    p = Process(target=train_loso, args=(val_idx, True))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
