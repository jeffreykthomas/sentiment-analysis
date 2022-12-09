import gradio as gr
import tensorflow as tf
import re
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


count_vect = pickle.load(open('countvect.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

from_disk = pickle.load(open('tv_layer.pkl', 'rb'))
text_vectorization = TextVectorization.from_config(from_disk['config'])
text_vectorization.set_weights(from_disk['weights'])

lr_model = pickle.load(open('logistic_model.pkl', 'rb'))
lstm_model = keras.models.load_model('lstm_model.h5')
bert_classifier_model = keras.models.load_model('bert_classifier.h5')


def get_bert_end_to_end(model):
    inputs_string = keras.Input(shape=(1,), dtype="string")
    indices = text_vectorization(inputs_string)
    outputs = model(indices)
    end_to_end_model = keras.Model(inputs_string, outputs, name="end_to_end_model")
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    end_to_end_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


bert_end_model = get_bert_end_to_end(bert_classifier_model)


def get_lr_results(text):
    sample_vec = count_vect.transform([text])
    return lr_model.predict(sample_vec)[0]


def get_lstm_results(text):
    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_tokens = pad_sequences(tokenized_text, maxlen=200)
    return lstm_model.predict(padded_tokens)[0][0]


def get_bert_results(text):
    return bert_end_model.predict([text])[0][0]


def decide(text):
    lr_result = get_lr_results(text)
    lstm_result = get_lstm_results(text)
    bert_result = get_bert_results(text)
    results = [
        lr_result,
        lstm_result,
        bert_result]
    if ((lr_result + lstm_result + bert_result) / 3) >= 0.6:
        return "Positive review (LR: {}, LSTM: {:.2}, BERT: {:.2}".format(*results)
    elif ((lr_result + lstm_result + bert_result) / 3) <= 0.4:
        return "Negative review (LR: {}, LSTM: {:.2}, BERT: {:.2}".format(*results)
    else:
        return "Neutral review (LR: {}, LSTM: {:.2}, BERT: {:.2}".format(*results)


example_sentence_1 = "I hate this toaster, they made no effort in making it. So cheap, it almost immediately broke!"
example_sentence_2 = "Great toaster! We love the way it toasted my bread so quickly. Very high quality components too."
example_sentence_3 = "Packaging was all torn and crushed. Planned on giving as Christmas gifts. Cheaply made " \
                     "material. Only flips one way. Terrible product!"
example_sentence_4 = "An epic undertaking and delivered with sophistication and style... " \
                     "an engaging and thought provoking read!"
example_sentence_5 = "Tried to bond a part of a foil that was damage but this adhesive is too weak in the bond it " \
                     "forms between these two materials. Will Crack upon any kind of force that gets applied even " \
                     "after letting it cure for a few days."
example_sentence_6 = "I really love this toothpaste. It does not have floride or xylitol. A big plus is my teeth feel " \
                     "cleaner with this toothpaste after brushing than with any other toothpaste I have ever had."
examples = [[example_sentence_1],
            [example_sentence_2],
            [example_sentence_3],
            [example_sentence_4],
            [example_sentence_5],
            [example_sentence_6]]

description = "Write out a product review to know the underlying sentiment."

gr.Interface(decide,
             inputs=gr.inputs.Textbox(lines=1, placeholder=None, default="", label=None),
             outputs='text',
             examples=examples,
             title="Sentiment analysis of product reviews",
             theme="grass", description=description,
             allow_flagging="auto",
             flagging_dir='flagging records').launch(enable_queue=True, inline=False)
