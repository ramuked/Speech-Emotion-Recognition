import sys
import os

import joblib

sys.path.append(os.path.dirname(__file__))
from settings import *
from scipy.io import wavfile
import librosa
from librosa import to_mono, resample
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import tensorflow as tf
import pandas as pd
import pickle
from pickle import dump
from pickle import load
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dropout,
    Activation,
)
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def extract_features(data):

    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


def create_model():
    model = Sequential()
    model.add(Conv1D(256, 8, padding="same", input_shape=(161, 1)))
    model.add(Activation("relu"))

    model.add(Conv1D(256, 8, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(128, 8, padding="same"))
    model.add(Activation("relu"))

    model.add(Conv1D(128, 8, padding="same"))
    model.add(Activation("relu"))

    model.add(Conv1D(128, 8, padding="same"))
    model.add(Activation("relu"))

    model.add(Conv1D(128, 8, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(64, 8, padding="same"))
    model.add(Activation("relu"))

    model.add(Conv1D(64, 8, padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(6))  # Target class number
    model.add(Activation("softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def get_features(path):
    data, sample_rate = librosa.load(path, duration=2, offset=0.6, sr=8025)

    res1 = extract_features(data)
    result = np.array(res1)

    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))

    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))
    return result


def predict(file_name):
    file_path = f"{MEDIA_DIR}/{file_name}"
    file_path_list = []
    file_path_list.append(file_path)
    model = create_model()
    model.load_weights(SAVED_WEIGHTS_PATH)
    scaler = joblib.load(SAVED_SCALER_PATH)
    encoder = joblib.load(SAVED_ENCODER_PATH)
    features = get_features(file_path)
    X_new = []
    for path in file_path_list:
        feature = get_features(path)
        for f in feature:
            X_new.append(f)
    Features_new = pd.DataFrame(X_new)
    X_new = Features_new.iloc[:, :-1].values
    X_new = scaler.fit_transform(X_new)
    X_new = np.expand_dims(X_new, axis=2)
    print(X_new.shape)
    prediction = model.predict(X_new)
    y_new = encoder.inverse_transform(prediction)
    print(model.summary())
    print(y_new)
    return y_new[0][0]
