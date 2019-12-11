
import tensorflow as tf

from argparse import ArgumentParser
from datetime import datetime
from time import time

from bert import run_classifier

from bert_classify_utils import load_from_folder, create_tokenizer, \
    model_fn_builder, get_estimator, getPrediction

from configs import config


class BERTClassifier(object):

    def __init__(self, project="imdb", mode="predict", skip_eval=True):
        try:
            self.vocab_file = config[project]['bert_config']['vocab_file']
            self.bert_config_file = config[project]['bert_config']['bert_config_file']
            self.init_checkpoint = config[project]['bert_config']['init_checkpoint']
            self.model_dir = config[project]['model_dir']
            self.train_folder = config[project]['train_folder']
            self.test_folder = config[project]['test_folder']
            self.train_batch_size = config[project]['train_params']['train_batch_size']
            self.predict_batch_size = config[project]['train_params']['predict_batch_size']
            self.max_seq_length = config[project]['train_params']['max_seq_length']
            self.learning_rate = config[project]['train_params']['learning_rate']
            self.num_train_epochs = config[project]['train_params']['num_train_epochs']
            self.warmup_proportion = config[project]['train_params']['warmup_proportion']
            self.save_checkpoints_steps = config[project]['train_params']['save_checkpoints_steps']
            self.save_summary_steps = config[project]['train_params']['save_summary_steps']
            self.labels = config[project]['data']['labels']
            self.data_column = config[project]['data']['data_column']
            self.label_column = config[project]['data']['label_column']
            self.estimator = None
        except:
            raise

        print("[INFO] Preparing Tokenizer...")
        self.tokenizer = create_tokenizer(vocab_file=self.vocab_file)
        print("[INFO] Done preparing tokenizer...\n")

        if mode == 'train':
            self.train(do_eval=(not skip_eval))
        else:
            try:
                self.model_checkpoint = config[project]['model_checkpoint']
            except:
                raise Exception("model_checkpoint is not defined in the configs file")
            self.estimator = get_estimator(
                self.model_checkpoint, self.bert_config_file, self.labels,
                self.model_dir, self.predict_batch_size)

    def train(self, do_eval=False):
        print("[INFO] Loading train data from folder...")
        train = load_from_folder(self.train_folder, labels=self.labels,
            data_column=self.data_column, label_column=self.label_column)
        print("[INFO] Done loading train data...\n")

        print("[INFO] Preparing train InputExample...")
        train_inputExamples = train.apply(
            lambda x: run_classifier.InputExample(
                guid=None,
                text_a=x[self.data_column],
                text_b=None,
                label=x[self.label_column]), axis=1)
        print("[INFO] Done preparing train InputExample...\n")

        label_list = list(range(len(self.labels)))
        print("[INFO] Preparing train features...")
        train_features = run_classifier.convert_examples_to_features(
            train_inputExamples, label_list, self.max_seq_length, self.tokenizer)
        print("[INFO] Done preparing train features...\n")

        train_input_fn = run_classifier.input_fn_builder(
            features=train_features,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=False)

        num_train_steps = int(len(train_features)/self.train_batch_size*self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        print(f"[INFO] No. of train steps: {num_train_steps}")
        print(f"[INFO] No. of warmup steps: {num_warmup_steps}")

        self.estimator = get_estimator(
            self.init_checkpoint, self.bert_config_file, self.labels,
            self.model_dir, self.train_batch_size,
            save_summary_steps=self.save_summary_steps,
            save_checkpoints_steps=self.save_checkpoints_steps,
            learning_rate=self.learning_rate, num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        print(f'[INFO] Beginning Training...!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print(f"[INFO] Training took time {datetime.now() - current_time} sec..!\n")

        if do_eval:
            evaluate()

    def evaluate(self):
        print("[INFO] Loading data from test folder...")
        test = load_from_folder(
            self.test_folder, labels=self.labels, data_column=self.data_column,
            label_column=self.label_column)
        print("[INFO] Done loading data...\n")

        print(f"[INFO] Started working on evaluation...")
        print("[INFO] Preparing test InputExample...")
        test_inputExamples = test.apply(
            lambda x: run_classifier.InputExample(
                guid=None,
                text_a=x[self.data_column],
                text_b=None,
                label=x[self.label_column]), axis=1)
        print("[INFO] Done preparing test InputExample...\n")

        label_list = list(range(len(LABELS)))
        print("[INFO] Preparing test features...")
        test_features = run_classifier.convert_examples_to_features(
            test_inputExamples, label_list, self.max_seq_length, self.tokenizer)
        print("[INFO] Done preparing test features...\n")

        test_input_fn = run_classifier.input_fn_builder(
            features=test_features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        print(f'[INFO] Beginning evaluating...!')
        self.estimator.evaluate(input_fn=test_input_fn, steps=None)
        print(f"[INFO] Done evaluating...\n")

    def predict(self, predict_file=None, text=None):
        if not (predict_file or text):
            raise Exception("Either of the predict_file or text should be provided")

        if predict_file:
            with open(predict_file) as f:
                pred_sentences = [line.strip() for line in f.readlines()]
        else:
            pred_sentences = [text.strip()]
        if len(pred_sentences) == 1:
            # Adding a dummy 2nd element so that the estimator does not throw Exception
            pred_sentences.append("")

        label_list = list(range(len(self.labels)))
        print(f"[INFO] Begin predicting...!")
        current_time = time()
        predictions = getPrediction(
            self.estimator, pred_sentences, labels=self.labels, label_list=label_list,
            max_seq_len=self.max_seq_length, tokenizer=self.tokenizer)
        if text:
            predictions = predictions[:1]
        print(f"[INFO] Predicting took {time() - current_time} secs...!\n")
        print(predictions)


if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action='store_true')
    group.add_argument("--predict", action='store_true')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--file", help="Path to the file with each line as a "
                                       "record to be classified")
    group2.add_argument("-t", "--text", help="Enter the text to classify")
    group2.add_argument("--skip-eval", action='store_true')
    args = parser.parse_args()

    if (args.skip_eval and args.predict) or (
            ((args.file or args.text) and args.train)):
        raise Exception("Invalid arguments combinations..!")


    if args.train:
        mode = "train"
    elif args.predict:
        if not(args.file or args.text):
            raise Exception("Either of `--predict-file` or `--text` "
                            "should be specified along with `--predict`")
        mode = 'predict'
    bc = BERTClassifier(project="imdb", mode=mode)
    bc.predict(args.file, args.text)
