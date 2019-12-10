
import tensorflow as tf

from argparse import ArgumentParser
from datetime import datetime

from bert import run_classifier

from bert_classify_utils import load_from_folder, create_tokenizer, \
    model_fn_builder, getPrediction


TRAIN_FOLDER = "imdb/train"
TEST_FOLDER = "imdb/test"
LABELS = ['pos', 'neg']
DATA_COLUMN = "sentence"
LABEL_COLUMN = "label"

OUTPUT_DIR = "output_dir/"

VOCAB_FILE = "BERT_BASE_DIR/vocab.txt"
BERT_CONFIG = "BERT_BASE_DIR/bert_config.json"
INIT_CHECKPOINT = "BERT_BASE_DIR/bert_model.ckpt"
# INIT_CHECKPOINT = "output_dir/model.ckpt-468"

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100


def get_estimator(num_train_steps=None, num_warmup_steps=None,
                  checkpoint=INIT_CHECKPOINT):
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
        bert_config_file=BERT_CONFIG,
        init_checkpoint=checkpoint,
        num_labels=len(LABELS),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})


def evaluate(tokenizer, estimator):
    print("[INFO] Loading data from folders...")
    test = load_from_folder(
        TEST_FOLDER, labels=LABELS, data_column=DATA_COLUMN,
        label_column=LABEL_COLUMN)
    print("[INFO] Done loading data...\n")

    test = test.sample(5000)

    print(f"[INFO] Started working on evaluation...")
    print("[INFO] Preparing test InputExample...")
    test_inputExamples = test.apply(
        lambda x: run_classifier.InputExample(
            guid=None,
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN]), axis=1)
    print("[INFO] Done preparing test InputExample...\n")

    label_list = list(range(len(LABELS)))
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


def train(tokenizer, do_eval=True, init_checkpoint=INIT_CHECKPOINT):
    print("[INFO] Loading data from folders...")
    train = load_from_folder(TRAIN_FOLDER, labels=LABELS,
        data_column=DATA_COLUMN, label_column=LABEL_COLUMN)
    print("[INFO] Done loading data...\n")

    train = train.sample(5000)

    print("[INFO] Preparing train InputExample...")
    train_inputExamples = train.apply(
        lambda x: run_classifier.InputExample(
            guid=None,
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN]), axis=1)
    print("[INFO] Done preparing train InputExample...\n")

    label_list = list(range(len(LABELS)))
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
        evaluate(tokenizer, estimator)


def predict(tokenizer, checkpoint, predict_file=None, text=None):
    if not (predict_file or text):
        raise Exception("Either of the predict_file or text should be provided")

    label_list = list(range(len(LABELS)))
    estimator = get_estimator(checkpoint=checkpoint)

    if predict_file:
        with open(predict_file) as f:
            pred_sentences = [line.strip() for line in f.readlines()]
    else:
        pred_sentences = [text.strip()]

    print(pred_sentences)

    print(f"[INFO] Beginning predicting...")
    predictions = getPrediction(
        estimator, pred_sentences, labels=LABELS, label_list=label_list,
        max_seq_len=MAX_SEQ_LENGTH, tokenizer=tokenizer)
    print(f"[INFO] Done predicting...\n")
    print(predictions)


def main(mode, skip_eval=False, predict_file=None, text=None,
         checkpoint=INIT_CHECKPOINT):

    print("[INFO] Preparing Tokenizer...")
    tokenizer = create_tokenizer(vocab_file=VOCAB_FILE)
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
    parser.add_argument("--init-checkpoint", help="Path to the checkpoint")
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
             checkpoint=args.init_checkpoint)
    elif args.predict:
        main(mode='predict', predict_file=args.predict_file, text=args.text,
             checkpoint=args.init_checkpoint)
