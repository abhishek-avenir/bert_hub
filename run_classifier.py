
import os
import pandas as pd
import tensorflow as tf

from argparse import ArgumentParser
from bert import modeling
from bert import optimization
from bert import run_classifier
from bert import tokenization
from datetime import datetime

OUTPUT_DIR = "OUTPUT_DIR_NAME"

if not os.path.exists(OUTPUT_DIR):
    tf.gfile.MakeDirs(OUTPUT_DIR)

DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'label'
LABELS = ['pos', 'neg']

TRAIN_FOLDER = "imdb/train"
TEST_FOLDER = "imdb/test"

VOCAB_FILE = "BERT_BASE_DIR/vocab.txt"
BERT_CONFIG = "BERT_BASE_DIR/bert_config.json"
INIT_CHECKPOINT = "BERT_BASE_DIR/bert_model.ckpt"

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

label_list = [i for i in range(len(LABELS))]


def load_from_folder(path, labels, data_column, label_column):
    i = 0
    data = {'sentence': [], 'label': []}
    for folder in labels:
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with tf.gfile.GFile(file_path, "r") as f:
                data[data_column].append(f.read())
                data[label_column].append(i)
        i += 1
    return pd.DataFrame.from_dict(data).reset_index(drop=True)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(
            tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if not is_training:
          return (None, predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


def model_fn_builder(bert_config, init_checkpoint, num_labels,
                     learning_rate, num_train_steps, num_warmup_steps):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
        is_training = not is_predicting

        (loss, predicted_labels, log_probs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, num_labels)

        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, _) = \
                modeling.get_assignment_map_from_checkpoint(
                    tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # TRAIN and EVAL
        if not is_predicting:
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps,
                use_tpu=False)

            # Calculate evaluation metrics. 
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels) 
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)  
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                ret = {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }
                # print(f"Result: {ret}")
                return ret

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, train_op=train_op)
            else:
                eval_metrics = metric_fn(label_ids, predicted_labels)
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        else:
            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    return model_fn


def get_estimator(num_train_steps=None, num_warmup_steps=None,
                  checkpoint=INIT_CHECKPOINT):
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(BERT_CONFIG),
        init_checkpoint=checkpoint,
        num_labels=len(LABELS),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})


def evaluate(tokenizer, estimator, sample=5000):
    print("[INFO] Loading data from test folder...")
    test = load_from_folder(
        TEST_FOLDER, labels=LABELS, data_column=DATA_COLUMN,
        label_column=LABEL_COLUMN)
    print("[INFO] Done loading data...\n")

    if sample:
        test = test.sample(sample)

    print("[INFO] Preparing test InputExample...")
    test_inputExamples = test.apply(
        lambda x: run_classifier.InputExample(
            guid=None,
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN]), axis=1)
    print("[INFO] Done preparing test InputExample...\n")

    print("[INFO] Preparing test features...")
    test_features = run_classifier.convert_examples_to_features(
        test_inputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print("[INFO] Done preparing test features...\n")

    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    print(f'[INFO] Beginning evaluating...!')
    estimator.evaluate(input_fn=test_input_fn, steps=None)
    print(f"[INFO] Done evaluating...\n")


def train(tokenizer, do_eval=True, init_checkpoint=INIT_CHECKPOINT, sample=5000):
    print("[INFO] Loading train data from folder...")
    train = load_from_folder(TRAIN_FOLDER, labels=LABELS,
        data_column=DATA_COLUMN, label_column=LABEL_COLUMN)
    print("[INFO] Done loading train data...\n")

    if sample:
        train = train.sample(sample)

    print("[INFO] Preparing train InputExample...")
    train_inputExamples = train.apply(
        lambda x: run_classifier.InputExample(
            guid=None,
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN]), axis=1)
    print("[INFO] Done preparing train InputExample...\n")

    print("[INFO] Preparing train features...")
    train_features = run_classifier.convert_examples_to_features(
        train_inputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print("[INFO] Done preparing train features...\n")

    train_input_fn = run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    print(f"[INFO] No. of train steps: {num_train_steps}")
    print(f"[INFO] No. of warmup steps: {num_warmup_steps}")
    estimator = get_estimator(num_train_steps, num_warmup_steps,
                              checkpoint=init_checkpoint)
    print(f'[INFO] Beginning Training...!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print(f"[INFO] Training took time {datetime.now() - current_time} sec..!\n")
    if do_eval:
        evaluate(tokenizer, estimator, sample=sample)

    def getPrediction(in_sentences):
        input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
        input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
        predictions = estimator.predict(predict_input_fn)
        return [(sentence, prediction['probabilities'], LABELS[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
    pred_sentences = [
      "That movie was absolutely awful",
      "The acting was a bit lacking",
      "The film was creative and surprising",
      "Absolutely fantastic!"]
    predictions = getPrediction(pred_sentences)
    print(predictions)


def getPrediction(estimator, in_sentences, labels, label_list, max_seq_len,
                  tokenizer):
    input_examples = [run_classifier.InputExample(
        guid="", text_a = x, text_b = None, label = 0) for x in in_sentences]
    input_features = run_classifier.convert_examples_to_features(
        input_examples, label_list, max_seq_len, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(
        features=input_features, seq_length=max_seq_len, is_training=False,
        drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    return [(sentence, prediction['probabilities'],
             labels[prediction['labels']])
            for sentence, prediction in zip(in_sentences, predictions)]


def predict(tokenizer, checkpoint, predict_file=None, text=None):
    if not (predict_file or text):
        raise Exception("Either of the predict_file or text should be provided")

    estimator = get_estimator(checkpoint=checkpoint)

    if predict_file:
        with open(predict_file) as f:
            pred_sentences = [line.strip() for line in f.readlines()]
    else:
        pred_sentences = [text.strip(), text.strip()]

    print(f"[INFO] Beginning predicting...")
    predictions = getPrediction(
        estimator, pred_sentences, labels=LABELS, label_list=label_list,
        max_seq_len=MAX_SEQ_LENGTH, tokenizer=tokenizer)
    print(f"[INFO] Done predicting...\n")
    print(predictions)


def main(mode, skip_eval=False, predict_file=None, text=None,
         checkpoint=INIT_CHECKPOINT):
    print("[INFO] Preparing Tokenizer...")
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)
    print("[INFO] Done preparing tokenizer...\n")
    if mode == 'train':
        train(tokenizer, do_eval=(not skip_eval), init_checkpoint=checkpoint)
    else:
        predict(tokenizer, checkpoint, predict_file, text)


if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action='store_true')
    group.add_argument("--predict", action='store_true')
    parser.add_argument("--checkpoint", help="Path to the checkpoint")
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "--predict-file",
        help="Path to the file with each line as a record to be classified")
    group2.add_argument("-t", "--text", help="Enter the text to classify")
    group2.add_argument("--skip-eval", action='store_true')
    args = parser.parse_args()

    if (args.skip_eval and args.predict) or (
            ((args.predict_file or args.text) and args.train)):
        raise Exception("Invalid arguments combinations..!")

    if args.train:
        main(mode='train', skip_eval=args.skip_eval,
             checkpoint=args.checkpoint)
    elif args.predict:
        if not args.checkpoint:
            raise Exception("Specify `--checkpoint` for `--predict`")
        if not(args.predict_file or args.text):
            raise Exception("Either of `--predict-file` or `--text` "
                            "should be specified along with `--predict`")
        main(mode='predict', predict_file=args.predict_file, text=args.text,
             checkpoint=args.checkpoint)
