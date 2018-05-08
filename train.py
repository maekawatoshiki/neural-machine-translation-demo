import os
import tensorflow as tf
import numpy as np
import zipfile
import random
import collections
import pickle
from six.moves.urllib.request import urlretrieve
import datetime
from itertools import chain

text = open("datasets_origin").read().split('\n')
text.pop()
print('Data size %d' % len(text))
print('data content: %s' % text)

train_text = []
test_text  = []
flg = False
for line in text:
    if line == '=====':
        flg = True
        continue
    if flg:
        test_text.append(line)
    else:
        train_text.append(line)
print(train_text)
print(test_text)

# Dictionary
vocabulary_size = 1000
def build_dictionary(words):
  count = collections.Counter(words).most_common(vocabulary_size - 2)
  dictionary = dict()
  dictionary['<PAD>'] = 0 
  dictionary['<UNK>'] = 1
  for word, _ in count:
      if word != '<UNK>':
          dictionary[word] = len(dictionary)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reverse_dictionary

if os.path.exists('dicts/dictionary14.pickle'):
    with open('dicts/dictionary14.pickle', 'rb') as handle:
        dictionary = pickle.load(handle)
    with open('dicts/reverse_dictionary14.pickle', 'rb') as handle:
        reverse_dictionary = pickle.load(handle)
else:
  words = list(chain.from_iterable([a.split() for a in train_text]))
  words += list(chain.from_iterable([a.split() for a in test_text]))
  words += list([a for a in "fgo rgo lgo bgo get".split(' ')])
  print("-"*99)
  print(words)
  dictionary, reverse_dictionary = build_dictionary(words)
  with open('dicts/dictionary14.pickle', 'wb') as handle:
      pickle.dump(dictionary, handle)
  with open('dicts/reverse_dictionary14.pickle', 'wb') as handle:
      pickle.dump(reverse_dictionary, handle)

# BatchGenerator
MAX_INPUT_SEQUENCE_LENGTH = 50
MAX_OUTPUT_SEQUENCE_LENGTH = 20

def gen_inst(s):
    pos = s
    x = int(pos[0])
    y = int(pos[1])
    res = ""
    t = x + y
    for i in random.sample(range(0, t), t):
        if i < x:
            res += "rgo "
        else:
            res += "fgo "
    res += "get"
    return res



# 26(alpha) + 10(numeric) + 1(space) + 1(period) = 38
dic_len = len(dictionary) 
PAD_ID = dic_len
GO_ID  = dic_len + 1
EOS_ID = dic_len + 2
NDS    = dic_len + 3

def char2limit(c):
    a = c.lower()
    if a.isalpha():
        return 0 + ord(a) - ord('a')
    elif a.isdigit():
        return ord(a) - ord('0') + 26
    elif a == ' ':
        return 36
    elif a == '.':
        return 37
    else:
        return '?'


class BatchGenerator(object):
    def __init__(self, text, batch_size):
        questions = []
        answers = []
        i = 0
        area = ""
        for t in text:
            if i < 5:
                area += t + " "
                i += 1
            elif i == 5:
                questions.append( [s.lower() for s in (area + t).split()] )
                area = ""
                i += 1
            elif i == 6:
                answers.append( [s.lower() for s in t.split()] )
                i = 0

        self._questions = questions
        self._answers   = answers

        self._batch_size = batch_size

    def next(self):
        input_sequences = list()
        encoder_inputs = list()
        decoder_inputs = list()
        labels = list()
        weights = list()

        for i in range(self._batch_size):
          choice = random.randint(0, len(self._questions) - 1)
          input_words = self._questions[choice]
          # print(input_words)
          input_word_ids = [word2id(word) for word in input_words]
          # print(input_words) 
          # reverse list and add padding
          reverse_input_word_ids = [0]*(MAX_INPUT_SEQUENCE_LENGTH-len(input_word_ids)) + input_word_ids[::-1]
          input_sequence = ' '.join(input_words)
          label_sequence = self._answers[choice]
          label_sequence = [s.lower() for s in gen_inst(label_sequence).split(' ')]
          # print(label_sequence)
          # input_word_ids = [word2id(word) for word in input_words]
          label_word_ids = [word2id(num) for num in label_sequence]
          # print("success")
          # print(label_word_ids)
          weight = [1.0]*len(label_word_ids)

          # append to lists
          input_sequences.append(input_sequence)
          encoder_inputs.append(reverse_input_word_ids)
          decoder_inputs.append([GO_ID] + label_word_ids + [PAD_ID]*(MAX_OUTPUT_SEQUENCE_LENGTH-len(label_word_ids)))
          labels.append(label_word_ids + [EOS_ID] + [PAD_ID]*(MAX_OUTPUT_SEQUENCE_LENGTH-len(label_word_ids)))
          weights.append(weight + [1.0] + [0.0]*((MAX_OUTPUT_SEQUENCE_LENGTH-len(weight))))

        return input_sequences, np.array(encoder_inputs).T, np.array(decoder_inputs).T, np.array(labels).T, np.array(weights).T

batch_size = 8
train_batches = BatchGenerator(train_text, batch_size)
test_batches = BatchGenerator(test_text, 10)

# Utils
def id2num(num_id):
  if num_id == PAD_ID:
      return 'P'
  if num_id == GO_ID:
      return 'G'
  if num_id == EOS_ID:
      return 'E'
  a = reverse_dictionary.get(num_id, "?")
  return a + ' '
  # return 'O'

def sampling(predictions):
    return ''.join([id2num(np.argmax(onehot[0])) for onehot in predictions])

def word2id(word):
    return dictionary.get(word, 0)

def id2word(id):
    return reverse_dictionary.get(id, "")

# Model
lstm_size = 256

def construct_graph(use_attention=True):
  encoder_inputs = list()
  decoder_inputs = list()
  labels = list()
  weights = list() 

  for _ in range(MAX_INPUT_SEQUENCE_LENGTH):
    encoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
  for _ in range(MAX_OUTPUT_SEQUENCE_LENGTH+1):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
    labels.append(tf.placeholder(tf.int32, shape=(None,)))
    weights.append(tf.placeholder(tf.float32, shape=(None,)))

  feed_previous = tf.placeholder(tf.bool)
  learning_rate = tf.placeholder(tf.float32)

    # Use LSTM cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
  with tf.variable_scope("seq2seq"):
      outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs,
              decoder_inputs,
              cell,
              vocabulary_size, # num_encoder_symbols
              NDS, # num_decoder_symbols
              128, # embedding_size
              feed_previous=feed_previous # False during training, True during testing
              )
      loss = tf.contrib.legacy_seq2seq.sequence_loss(outputs, labels, weights) 
  predictions = tf.stack([tf.nn.relu(output) for output in outputs])

  tf.summary.scalar('learning rate', learning_rate)
  tf.summary.scalar('loss', loss)
  merged = tf.summary.merge_all()

  return encoder_inputs, decoder_inputs, labels, weights, learning_rate, feed_previous, outputs, states, loss, predictions, merged

encoder_inputs, decoder_inputs, labels, weights, learning_rate, feed_previous, outputs, states, loss, predictions, merged = construct_graph()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()

# Run session
today_dt = datetime.date.today()
today = today_dt.strftime("%Y%m%d")

with tf.Session() as sess:
  saver.restore(sess, "checkpoints/20180426_model-65000steps.ckpt")
        # 20180426_model-65000steps.ckpt.inde
  # sess.run(tf.global_variables_initializer())
  current_learning_rate = 0.02

  for step in range(500001):
    feed_dict = dict()
    current_train_sequences, current_train_encoder_inputs, current_train_decoder_inputs, current_train_labels, current_weights = train_batches.next()
    feed_dict = {encoder_inputs[i]: current_train_encoder_inputs[i] for i in range(MAX_INPUT_SEQUENCE_LENGTH)}
    feed_dict.update({decoder_inputs[i]: current_train_decoder_inputs[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
    feed_dict.update({labels[i]: current_train_labels[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
    feed_dict.update({weights[i]: current_weights[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
    feed_dict.update({feed_previous: False})

    if step != 0 and step % 50000 == 0:
        current_learning_rate /= 2
    feed_dict.update({learning_rate: current_learning_rate})

    _, current_train_loss, current_train_predictions, train_summary = sess.run([optimizer, loss, predictions, merged], feed_dict=feed_dict)

    if step % 20 == 0:
      print('Step %d:' % step)
      print('Training set:')
      print('  Loss       : ', current_train_loss)
      print('  Input            : ', current_train_sequences[0])
      print('  Correct output   : ', ''.join([id2num(n) for n in current_train_labels.T[0]]))
      print('  Generated output : ', sampling(current_train_predictions))

      test_feed_dict = dict() 
      current_test_sequences, current_test_encoder_inputs, current_test_decoder_inputs, current_test_labels, current_test_weights = test_batches.next()
      test_feed_dict = {encoder_inputs[i]: current_test_encoder_inputs[i] for i in range(MAX_INPUT_SEQUENCE_LENGTH)}
      test_feed_dict.update({decoder_inputs[i]: current_test_decoder_inputs[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
      test_feed_dict.update({labels[i]: current_test_labels[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
      test_feed_dict.update({weights[i]: current_test_weights[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})

      test_feed_dict.update({feed_previous: True})
      test_feed_dict.update({learning_rate: current_learning_rate})
      current_test_loss, current_test_predictions, test_summary = sess.run([loss, predictions, merged], feed_dict=test_feed_dict)

      print('Test set:')
      print('  Loss       : ', current_test_loss)
      print('  Input            : ', current_test_sequences[0])
      print('  Correct output   : ', ''.join([id2num(n) for n in current_test_labels.T[0]]))
      print('  Generated output : ', sampling(current_test_predictions))
      print('='*50)

    if step % 5000 == 0:
        # Save the variables to disk.
        save_path = saver.save(sess, "checkpoints/{}_model-{}steps.ckpt".format(today, step))
        print("Model saved in file: %s" % save_path)
