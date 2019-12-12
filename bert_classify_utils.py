
import os
import pandas as pd
import re
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from bert import modeling
from bert import optimization
from bert import run_classifier
from bert import tokenization


def transform_labels(df, le=None):
    if not le:
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
    else:
        df['label'] = le.transform(df['label'])
    return df.reset_index(drop=True), le


def load_dataset(csv_file, le=None):
    df = pd.read_csv(csv_file)
    return transform_labels(df, le)


def split_into_train_test(csv_file, stratify=True, **kwargs):
    dirname = os.path.dirname(os.path.abspath(csv_file))
    df = pd.read_csv(csv_file)
    X = df['sentence']
    y = df['label']
    stratify_col = None
    if stratify:
        stratify_col = y
    train_X, test_X, train_y, test_y = \
        train_test_split(X, y, stratify=stratify_col, **kwargs)
    train = pd.concat([train_X, train_y], axis=1).reset_index(drop=True)
    test = pd.concat([test_X, test_y], axis=1).reset_index(drop=True)
    train.to_csv(os.path.join(dirname, 'train.csv'), index=False)
    test.to_csv(os.path.join(dirname, 'test.csv'), index=False)
    return train, test


def load_from_folder(path, le=None):
    data = {'sentence': [], 'label': []}
    for folder in os.path.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with tf.gfile.GFile(file_path, "r") as f:
                data['sentence'].append(f.read())
                data['label'].append(folder)
    df = pd.DataFrame.from_dict(data)
    return transform_labels(df, le)


def create_tokenizer(vocab_file, do_lower_case=True):
    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


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
        probabilities = tf.nn.softmax(logits, axis=-1)
        predictions = tf.argmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(
            tf.argmax(log_probs, axis=-1, output_type=tf.int32))

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (per_example_loss, loss, predicted_labels, log_probs)


def model_fn_builder(bert_config_file, init_checkpoint, num_labels,
                     learning_rate, num_train_steps, num_warmup_steps):
    bert_config=modeling.BertConfig.from_json_file(bert_config_file)
    def model_fn(features, labels, mode, params):

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (per_example_loss, loss, predicted_labels, log_probs) = \
            create_model(bert_config,
                is_training, input_ids, input_mask, segment_ids,
                label_ids, num_labels)

        (assignment_map, _) = \
            modeling.get_assignment_map_from_checkpoint(
                tf.trainable_variables(), init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps,
                use_tpu=False)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # Calculate evaluation metrics. 

            def metric_fn(per_example_loss, label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                loss = tf.metrics.mean(values=per_example_loss)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }
            # def metric_fn(label_ids, predicted_labels):
            #     accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            #     f1_score = tf.contrib.metrics.f1_score(
            #         label_ids,
            #         predicted_labels)
            #     auc = tf.metrics.auc(
            #         label_ids,
            #         predicted_labels)
            #     recall = tf.metrics.recall(
            #         label_ids,
            #         predicted_labels)
            #     precision = tf.metrics.precision(
            #         label_ids,
            #         predicted_labels) 
            #     true_pos = tf.metrics.true_positives(
            #         label_ids,
            #         predicted_labels)
            #     true_neg = tf.metrics.true_negatives(
            #         label_ids,
            #         predicted_labels)
            #     false_pos = tf.metrics.false_positives(
            #         label_ids,
            #         predicted_labels)  
            #     false_neg = tf.metrics.false_negatives(
            #         label_ids,
            #         predicted_labels)
            #     ret = {
            #         "eval_accuracy": accuracy,
            #         "f1_score": f1_score,
            #         # "auc": auc,
            #         "precision": precision,
            #         "recall": recall,
            #         "true_positives": true_pos,
            #         "true_negatives": true_neg,
            #         "false_positives": false_pos,
            #         "false_negatives": false_neg
            #     }
            #     return ret
            eval_metrics = \
                metric_fn(per_example_loss, label_ids, predicted_labels)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        else:
            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    return model_fn


def get_estimator(checkpoint, bert_config_file, labels, batch_size, output_dir=None,
                  save_summary_steps=None, save_checkpoints_steps=None,
                  learning_rate=None, num_train_steps=None, num_warmup_steps=None):
    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=save_checkpoints_steps)

    model_fn = model_fn_builder(
        bert_config_file=bert_config_file,
        init_checkpoint=checkpoint,
        num_labels=len(labels),
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": batch_size})


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
    return [(sentence, pred['probabilities'], labels[pred['labels']])
            for sentence, pred in zip(in_sentences, predictions)]
