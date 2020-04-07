from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tokenization
import tensorflow as tf
import CjPreprocessor
from flask import Flask, make_response, jsonify, request
import json
import numpy as np
import queue

flags = tf.flags
FLAGS = flags.FLAGS

rootPath = os.path.dirname(os.path.realpath(__file__))
flags.DEFINE_integer("bert_pool", 3, "")

flags.DEFINE_string("vocab_file", rootPath + "/conf/vocab.txt", "")
flags.DEFINE_string("model_dir", rootPath + "/exported_model/1586176288", "")
flags.DEFINE_bool("do_lower_case", False, "")
flags.DEFINE_integer("max_seq_length", 196, "")
flags.DEFINE_string("host", "127.0.0.1", "")
app = Flask (__name__)
graph = tf.Graph()
bertServicePool = None

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    # tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    # tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    # tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    # tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

class BertServicePool():
  def __init__(self, poolCnt = FLAGS.bert_pool):
    # 서비스 N개 만들어서 풀에 넣기
    q = queue.Queue()
    for i in range(0, poolCnt):
      wk = BertWocker(i)
      q.put(wk)
    self.q = q

  def Predict(self, text):
    try:
      wk = self.q.get()
      d = wk.Predict(text)
      return d
    except Exception as e:
      print("Predict error")
      print(e)
      return
    finally:
      self.q.put(wk)
    return

class BertWocker():
  def __init__(self, wokerId):
    self.id = wokerId
    print("init BertWocker : " + str(wokerId))
    self.tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    self.label_list = ["0", "1"]

    graph = tf.Graph()
    with graph.as_default():
      self.sess = tf.Session(graph=graph)
      tf.saved_model.loader.load(self.sess, ['serve'], FLAGS.model_dir)
      self.tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
      self.tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
      self.tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
      self.tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
      self.tensor_outputs = graph.get_tensor_by_name('loss/Softmax:0')

  def Predict(self, text):
    examples = []
    examples.append(InputExample(guid="predict-0", text_a=tokenization.convert_to_unicode(text), text_b=None, label=tokenization.convert_to_unicode("0")))
    features = convert_examples_to_features(examples, self.label_list,
                                            FLAGS.max_seq_length, self.tokenizer)
    feature = features[0]

    input_ids = feature.input_ids
    input_mask = feature.input_mask
    label_ids = [feature.label_id]
    segment_ids = feature.segment_ids
    result = self.sess.run(self.tensor_outputs, feed_dict={
          self.tensor_input_ids: np.array(input_ids).reshape(-1, FLAGS.max_seq_length),
          self.tensor_input_mask: np.array(input_mask).reshape(-1, FLAGS.max_seq_length),
          self.tensor_label_ids: np.array(label_ids),
          self.tensor_segment_ids: np.array(segment_ids).reshape(-1, FLAGS.max_seq_length),
    })

    answer = 0
    prediction = result[0]
    # prediction[1] == 1 is yok
    if prediction[0] < prediction[1]:
      answer = 1

    print("***** Service results *****")
    print("text : {}".format(text))
    print("work Id : {}".format(self.id))
    print("string : {}".format(text))
    print("prediction : {}, {}".format(prediction[0], prediction[1]))
    print("answer : {}".format(answer))

    data = {'code': 0, 'answer': answer, 'left': str(prediction[0]), 'right': str(prediction[1])}

    return data

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def hello_predict():
    global bertServicePool

    print(request.json['targetString'])

    text = CjPreprocessor.PreprocessString(request.json['targetString'])
    data = bertServicePool.Predict(text)
    json_data = json.dumps(data, ensure_ascii=False)
    res = make_response(json_data)
    
    res.headers['Content-Type'] = 'application/json'

    return res

def ApiServer():
  tf.logging.set_verbosity(tf.logging.ERROR)
  app.run(host=FLAGS.host)
  return

if __name__ == "__main__":
  bertServicePool = BertServicePool()
  ApiServer()