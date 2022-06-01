import re
import os
import unicodedata
import urllib3
import zipfile
import shutil
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


# google 공유파일 불러오기
# import gdown
# google_path = 'https://drive.google.com/uc?id='
# file_id = '1AVHhyj1_t4nhHnLZPlfZYZs3y_gIsMuW'
# output_name = 'test.xlsx'
# gdown.download(google_path+file_id,output_name,quiet=False)

# 불러온 파일 읽기
import pandas as pd

df = pd.read_excel('D:\Image_project/test.xlsx')

kor = df['원문'].iloc[1:200000:2]
eng = df['번역문'].iloc[1:200000:2]

def preprocess_sentence(sent):

  # 단어와 구두점 사이에 공백을 만듭니다.
  # Ex) "he is a boy." => "he is a boy ."
  sent = re.sub(r"([?.!,¿])", r" \1", sent)

  # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
  sent = re.sub(r"[^a-zA-Z가-힣!.?]+", r" ", sent)

  # 다수 개의 공백을 하나의 공백으로 치환
  sent = re.sub(r"\s+", " ", sent)
  return sent

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
hidden_units = 256  # Latent dimensionality of the encoding space.
embedding_dim = 64
num_samples = 100000  # Number of samples to train on.


def load_preprocessed_data():
  encoder_input, decoder_input, decoder_target = [], [], []

  for k, e in zip(kor, eng):
    # line_index, src_line, tar_line = line.strip().split('\t')

    src_line = e
    tar_line = k
    src_line = [w for w in src_line.split()]

    # target 데이터 전처리
    tar_line = preprocess_sentence(tar_line)
    tar_line_in = [w for w in ("<sos> " + tar_line).split()]  # teacher forcing을 위한 정답셋
    tar_line_out = [w for w in (tar_line + " <eos>").split()]

    encoder_input.append(src_line)
    decoder_input.append(tar_line_in)
    decoder_target.append(tar_line_out)

  return encoder_input, decoder_input, decoder_target

sents_eng_in, sents_kor_in, sents_kor_out  = load_preprocessed_data()

# 영어 원문

tokenizer_en = Tokenizer(filters='',lower = False)
tokenizer_en.fit_on_texts(sents_eng_in)

encoder_input = tokenizer_en.texts_to_sequences(sents_eng_in)

# 한국어
tokenizer_kor = Tokenizer(filters='',lower = False)
tokenizer_kor.fit_on_texts(sents_kor_in)
tokenizer_kor.fit_on_texts(sents_kor_out)


decoder_input = tokenizer_kor.texts_to_sequences(sents_kor_in)
decoder_target = tokenizer_kor.texts_to_sequences(sents_kor_out)

encoder_input = pad_sequences(encoder_input,padding = 'post')
decoder_input = pad_sequences(decoder_input,padding = 'post')
decoder_target = pad_sequences(decoder_target,padding = 'post')

# 단어길이
src_vocab_size = len(tokenizer_en.word_index) + 1
tar_vocab_size = len(tokenizer_kor.word_index) + 1

src_to_index = tokenizer_en.word_index  # word : idx
index_to_src = tokenizer_en.index_word  # idx : word
tar_to_index = tokenizer_kor.word_index # word : idx
index_to_tar = tokenizer_kor.index_word # idx : word

indices = np.arange(encoder_input.shape[0])
print(indices)
np.random.shuffle(indices)
print('랜덤 시퀀스 :',indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

n_of_val = int(num_samples*0.1)

encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

import tensorflow
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model
import tensorflow as tf

# 인코더
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(src_vocab_size, embedding_dim)(encoder_inputs) # 임베딩 층

###############
encoder_lstm = LSTM(hidden_units, return_state=True , return_sequences=True) # 상태값 리턴을 위해 return_state는 True
###############

encoder_outputs, state_h, state_c = encoder_lstm(enc_emb) # 은닉 상태와 셀 상태를 리턴

encoder_states = [state_h, state_c] # 인코더의 은닉 상태와 셀 상태를 저장

from keras.layers import AdditiveAttention

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
      super().__init__()

      self.W1 = tf.keras.layers.Dense(units, use_bias = False) # W1@ht
      self.W2 = tf.keras.layers.Dense(units, use_bias = False) # W2@hs

      self.attention = tf.keras.layers.AdditiveAttention()

  def call(self, query, value):

      # W1 @ ht
      w1_query = self.W1(query)

      # W2 @ hs
      w2_key = self.W2(value)

      # attention
      context_vector, attention_weights = self.attention(inputs = [w1_query, value, w2_key], return_attention_scores = True)

      return context_vector, attention_weights


# 디코더
decoder_inputs = Input(shape=(None,))

# 임베딩 층
dec_emb_layer = Embedding(tar_vocab_size, hidden_units)

# 임베딩 결과
dec_emb = dec_emb_layer(decoder_inputs)

######################
# 상태값 리턴을 위해 return_state는 True, 모든 시점에 대해서 단어를 예측하기 위해 return_sequences는 True
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
######################

# 인코더의 은닉 상태를 초기 은닉 상태(initial_state)로 사용
decoder_outputs, _ , _ = decoder_lstm(dec_emb,initial_state=encoder_states)

######################
# attention
S_ = tf.concat([state_h[:, tf.newaxis, :], decoder_outputs[:, :-1, :]], axis=1)

attention = BahdanauAttention(hidden_units)
context_vector, _ = attention(S_, encoder_outputs)

concat = tf.concat([decoder_outputs, context_vector], axis=-1)

# 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
decoder_dense = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(concat)

######################

# 모델의 입력과 출력을 정의.
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

adam = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer=adam)

model.summary()
