
import tensorflow as tf

from argparse import ArgumentParser
from datetime import datetime
from time import time

from bert import run_classifier

from bert_classify_utils import load_from_folder, create_tokenizer, \
    model_fn_builder, get_estimator, getPrediction

from configs import config



class BERTClassifier(object):

    def __init__(project="imdb"):
        self.VOCAB_FILE = config[project]['bert_config']['vocab_file']
        self.BERT_CONFIG = config[project]['bert_config']['bert_config_file']
        self.INIT_CHECKPOINT = config[project]['bert_config']['init_checkpoint']
        self.MODEL_DIR = config[project]['model_dir']
        self.TRAIN_FOLDER = config[project]['train_folder']
        self.TEST_FOLDER = config[project]['test_folder']
        self.TRAIN_BATCH_SIZE = config[project]['train_params']['train_batch_size']
        self.PREDICT_BATCH_SIZE = config[project]['train_params']['predict_batch_size']
        self.MAX_SEQ_LENGTH = config[project]['train_params']['max_seq_length']
        self.LEARNING_RATE = config[project]['train_params']['learning_rate']
        self.NUM_TRAIN_EPOCHS = config[project]['train_params']['num_train_epochs']
        self.WARMUP_PROPORTION = config[project]['train_params']['warmup_proportion']
        self.SAVE_CHECKPOINTS_STEPS = config[project]['train_params']['save_checkpoints_steps']
        self.SAVE_SUMMARY_STEPS = config[project]['train_params']['save_summary_steps']
        self.LABELS = config[project]['data']['labels']
        self.DATA_COLUMN = config[project]['data']['data_column']
        self.LABEL_COLUMN = config[project]['data']['label_column']


def evaluate(tokenizer, estimator):
    print("[INFO] Loading data from test folder...")
    test = load_from_folder(
        TEST_FOLDER, labels=LABELS, data_column=DATA_COLUMN,
        label_column=LABEL_COLUMN)
    print("[INFO] Done loading data...\n")

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


def train(tokenizer, do_eval=True, init_checkpoint=INIT_CHECKPOINT,
          batch_size=TRAIN_BATCH_SIZE, bert_config_file=BERT_CONFIG,
          labels=LABELS, output_dir=MODEL_DIR, save_summary_steps=SAVE_SUMMARY_STEPS,
          save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS, learning_rate=LEARNING_RATE):
    print("[INFO] Loading train data from folder...")
    train = load_from_folder(TRAIN_FOLDER, labels=LABELS,
        data_column=DATA_COLUMN, label_column=LABEL_COLUMN)
    print("[INFO] Done loading train data...\n")

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

    num_train_steps = int(len(train_features)/TRAIN_BATCH_SIZE*NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    print(f"[INFO] No. of train steps: {num_train_steps}")
    print(f"[INFO] No. of warmup steps: {num_warmup_steps}")

    estimator = get_estimator(
        init_checkpoint, bert_config_file, labels, output_dir, batch_size,
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        learning_rate=learning_rate, num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    print(f'[INFO] Beginning Training...!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print(f"[INFO] Training took time {datetime.now() - current_time} sec..!\n")

    if do_eval:
        evaluate(tokenizer, estimator)


def predict(tokenizer, checkpoint, predict_file=None, text=None,
            bert_config_file=BERT_CONFIG, labels=LABELS, output_dir=MODEL_DIR, 
            batch_size=PREDICT_BATCH_SIZE, max_seq_len=MAX_SEQ_LENGTH):
    if not (predict_file or text):
        raise Exception("Either of the predict_file or text should be provided")

    label_list = list(range(len(labels)))
    estimator = get_estimator(
        checkpoint, bert_config_file, labels, output_dir, batch_size)

    if predict_file:
        with open(predict_file) as f:
            pred_sentences = [line.strip() for line in f.readlines()]
    else:
        pred_sentences = [text.strip()]
    if len(pred_sentences) == 1:
        # Adding a dummy 2nd element so that the estimator does not throw Exception
        pred_sentences.append("")

    print(f"[INFO] Begin predicting...!")
    current_time = time()
    predictions = getPrediction(
        estimator, pred_sentences, labels=labels, label_list=label_list,
        max_seq_len=max_seq_len, tokenizer=tokenizer)
    if text:
        predictions = predictions[:1]
    print(f"[INFO] Predicting took {time() - current_time} secs...!\n")
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

    set_global_vars()
    
    if args.train:
        main(mode='train', skip_eval=args.skip_eval,
             checkpoint=args.checkpoint)
    elif args.predict:
        if not args.checkpoint:
            raise Exception("`--checkpoint` is required for `--predict`")
        if not(args.predict_file or args.text):
            raise Exception("Either of `--predict-file` or `--text` "
                            "should be specified along with `--predict`")
        main(mode='predict', predict_file=args.predict_file, text=args.text,
             checkpoint=args.checkpoint)
