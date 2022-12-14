import re
import numpy as np
import pickle

from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--algo", type=str, help="lstm/logistic")

algo = ap.parse_args().algo

train_file = 'data/train.ft.txt'
test_file = 'data/test.ft.txt'


def prepare_dataset():
    with open(train_file) as file:
        train_file_lines = [line.rstrip() for line in file]
    with open(test_file) as file:
        test_file_lines = [line.rstrip() for line in file]
    return train_file_lines, test_file_lines


def clean_data(data_file):
    labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in data_file]
    labels = np.array(labels)
    sentences = [x.split(' ', 1)[1][:-1].lower() for x in data_file]
    data_length = len(sentences) / 10

    for i in range(len(data_file)):
        sentences[i] = re.sub('\d', '0', sentences[i])

    for i in range(len(sentences)):
        if 'www.' in sentences[i] or 'http:' in sentences[i] or 'https:' in sentences[i] or '.com' in \
                sentences[i]:
            sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", sentences[i])

        if i % data_length == 0:
            print('--' + str(int(i / data_length) * 10) + '%', end='')

    print('--100%--Done--')
    return labels, sentences


def clean_texts(texts):
    labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in texts]
    labels = np.array(labels)
    sentences = [x.split(' ', 1)[1][:-1].lower() for x in texts]
    stwords = stopwords.words('english')
    l = len(sentences) / 10
    temp_texts = []
    for i in range(len(sentences)):
        text = re.sub('\d', '0', sentences[i])
        if 'www.' in text or 'http:' in text or 'https:' in text or '.com' in text:  # remove links and urls
            text = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", " ", text)

        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [word for word in text if not word in stwords]  # remove stopwords
        text = ' '.join(text)
        temp_texts.append(text)
        if i % l == 0:
            print('--' + str(int(i / l) * 10) + '%', end='')
    print('--100%--Done !')
    return labels, temp_texts


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


def tokenize(train_sentences, test_sentences):
    # Load text vectorization layer trained with BERT.py
    from_disk = pickle.load(open('tv_layer.pkl', 'rb'))
    text_vectorization = TextVectorization.from_config(from_disk['config'])
    text_vectorization.set_weights(from_disk['weights'])

    tokenized_train_sentences = text_vectorization(train_sentences).numpy()
    tokenized_test_sentences = text_vectorization(test_sentences).numpy()

    return tokenized_train_sentences, tokenized_test_sentences


def lstm():
    train_file_lines, test_file_lines = prepare_dataset()

    print('Processing Training data')
    y_train, train_sentences = clean_data(train_file_lines)
    print('\nProcessing Test data')
    y_test, test_sentences = clean_data(test_file_lines)
    print('Fitting data...')
    x_train, x_test = tokenize(train_sentences, test_sentences)

    max_features = 20000  # Only consider the top 20k words

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    print(len(x_train), "Training sequences")
    print(len(x_test), "Validation sequences")

    weight_path = "lstm_weights_2.hdf5"
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks = [checkpoint, early_stopping]

    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=callbacks)
    return model


def logistic_regression():
    train_file_lines, test_file_lines = prepare_dataset()
    print('Processing Training data')
    y_train, train_texts = clean_texts(train_file_lines)
    print('\nProcessing Test data')
    y_test, test_texts = clean_texts(test_file_lines)

    print('Fitting data...')
    count_vect = CountVectorizer()
    count_vect.fit(train_texts)
    print('fit complete !')

    print('Transforming training set...')
    x_train = count_vect.transform(train_texts)

    print('Transforming test set...')
    x_test = count_vect.transform(test_texts)

    lr_model = LogisticRegression(n_jobs=-1, max_iter=150)
    lr_model.fit(x_train, y_train)

    pred_lr = lr_model.predict(x_test)
    print('Accuracy:', accuracy_score(y_test, pred_lr))

    pickle.dump(lr_model, open('logistic_model.pkl', 'wb'))
    pickle.dump(count_vect, open('countvect.pkl', 'wb'))


def save_results(model_history, save_dir):
    # save progress and charts
    print('Saving results')
    epoch = model_history.epoch
    accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    sns.set()
    fig = plt.figure(0, (12, 4))

    ax = plt.subplot(1, 2, 1)
    sns.lineplot(x=epoch, y=accuracy, label='train')
    sns.lineplot(x=epoch, y=val_accuracy, label='valid')
    plt.title('Accuracy')
    plt.tight_layout()

    ax = plt.subplot(1, 2, 2)
    sns.lineplot(x=epoch, y=loss, label='train')
    sns.lineplot(x=epoch, y=val_loss, label='valid')
    plt.title('Loss')
    plt.tight_layout()

    plt.savefig(save_dir + '/epoch_history.png')
    plt.close(fig)


if __name__ == '__main__':
    if algo == 'logistic':
        logistic_regression()
    else:
        lstm_model = lstm()
        save_results(lstm_model.history, 'results')
