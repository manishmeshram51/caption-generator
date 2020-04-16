root_path = './images/'
wpath = './models/'

import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
L = keras.layers
K = keras.backend
import utils
import time
import zipfile
import json
from collections import defaultdict
import re
import random
from random import choice
import os
from keras_utils import reset_tf_session
import tqdm_utils


# Extract image features
IMG_SIZE = 299

# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model

# Define architecture
IMG_EMBED_SIZE = 2048
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = 1

s = reset_tf_session()
tf.set_random_seed(42)

#decoder
class decoder:

    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
    
    sentences = tf.placeholder('int32', [None, None])

    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')

    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
    
    word_embed = L.Embedding(8769, WORD_EMBED_SIZE)
 
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
     
    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
  
    token_logits = L.Dense(8769,
                           input_shape=(None, LOGIT_BOTTLENECK))
    
    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds)) 

    word_embeds = word_embed(sentences[:, :-1]) 
    
    hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))
    
    flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS]) 

    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states)) 

    flat_ground_truth = tf.reshape(sentences[:, 1:], [-1])

    flat_loss_mask = tf.not_equal(flat_ground_truth, pad_idx)

    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth, 
        logits=flat_token_logits
    )

    loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask))

# define optimizer operation to minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(decoder.loss)

# will be used to save/load network weights.
saver = tf.train.Saver()

# intialize all variables
s.run(tf.global_variables_initializer())

# will be used to save/load network weights.
saver = tf.train.import_meta_graph(wpath + 'weights_11.meta')


class final_model:
    # CNN encoder 
    encoder, preprocess_for_model = get_cnn_encoder()

    saver.restore(s, os.path.abspath(wpath + "weights_11"))
    
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")

    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

    img_embeds = encoder(input_images)

    init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)

    current_word = tf.placeholder('int32', [1], name='current_input')

    word_embed = decoder.word_embed(current_word)

    new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))

    new_probs = tf.nn.softmax(new_logits)

    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)


# _ = np.array([0.5, 0.4, 0.1])
# for t in [0.01, 0.1, 1, 10, 100]:
#     print(" ".join(map(str, _**(1/t) / np.sum(_**(1/t)))), "with temperature", t)
# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

import pickle
vocab ={}
with open('models/vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)
vocab_inverse = {idx: w for w, idx in vocab.items()}
# this is an actual prediction loop
def generate_caption(image, t=1, sample=False, max_len=20):
    # condition lstm on the image
    s.run(final_model.init_lstm, 
          {final_model.input_images: [image]})
    
    caption = [vocab[START]]
    
    for _ in range(max_len):
        next_word_probs = s.run(final_model.one_step, 
                                {final_model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()
        
        next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break
       
    return list(map(vocab_inverse.get, caption))


# look at validation prediction example
def apply_model_to_image_raw_bytes(img):
    # img = utils.decode_image_from_buf(raw)
    # fig = plt.figure(figsize=(7, 7))
    # plt.grid('off')
    # plt.axis('off')
    # plt.imshow(img)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    # print(' '.join(generate_caption(img)[1:-1]))
    return str(' '.join(generate_caption(img)[1:-1]))
    # plt.show()


# print(apply_model_to_image_raw_bytes(open(root_path + "street.jpg", "rb").read()))
