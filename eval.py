#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import argparse
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data loading params
args = argparse.ArgumentParser()
args.add_argument("--dev_sample_percentage", type=float, default=.1, help="Percentage of the training data to use for validation")
args.add_argument("--positive_data_file", type=str, default="./data/rt-polaritydata/rt-polarity.pos", help="Data source for the positive data.")
args.add_argument("--negative_data_file", type=str, default="./data/rt-polaritydata/rt-polarity.neg", help="Data source for the negative data.")

# Model Hyperparameters
args.add_argument("--embedding_dim", type=int, default=128, help="Dimensionality of character embedding (default: 128)")
args.add_argument("--filter_sizes", type=str, default="3,4,5", help="Comma-separated filter sizes (default: '3,4,5')")
args.add_argument("--num_filters", type=int, default=128, help="Number of filters per filter size (default:128)")
args.add_argument("--dropout_keep_prob", type=float, default=.5, help="Dropout keep probability (default: 0.5)")
args.add_argument("--l2_reg_lambda", type=float, default=.0, help="L2 regularization labda (default: 0.0)")

# Eval Parameters
args.add_argument("--batch_size", type=int, default=64, help="Batch Size (default: 64)")
args.add_argument("--checkpoint_dir", type=str, default="", help="Checkpoint directory from training run")
args.add_argument("--eval_train", type=bool, default=False, help="Evaluate on all training data")

# Misc Parameters
args.add_argument("--allow_soft_placement", type=bool, default=True, help="Allow device soft device placement")
args.add_argument("--log_device_placement", type=bool, default=False, help="Log placement of ops on devices")

FLAGS = args.parse_args()

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ===========================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
