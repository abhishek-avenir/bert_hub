
import numpy as np
import os
import tensorflow as tf
import pickle

from argparse import ArgumentParser
from datetime import datetime
from time import time

from bert import run_classifier

from bert_classify_utils import load_dataset, load_from_folder, \
    create_tokenizer, model_fn_builder, get_estimator, getPrediction

from config import PARAMS


class BERTClassifier(object):

    def __init__(self, project, mode, skip_eval=False):
        try:
            params = PARAMS[project]
            self.vocab_file = params['vocab_file']
            self.bert_config_file = params['bert_config_file']
            self.max_seq_length = params['max_seq_length']
            self.label_encoder_pkl = params['label_encoder']
            self.estimator = None
        except:
            raise

        print("[INFO] Preparing Tokenizer...")
        self.tokenizer = create_tokenizer(vocab_file=self.vocab_file)
        print("[INFO] Done preparing tokenizer...\n")

        if mode == 'train':
            self.set_train_test_params(params)
            self.do_train()
            if not skip_eval and not self.test.empty:
                self.do_eval()
        else:
            try:
                predict_params = params['predict_params']
                self.model_checkpoint = predict_params['model_checkpoint']
                self.predict_batch_size = predict_params['predict_batch_size']
            except:
                raise
            with open(self.label_encoder_pkl, 'rb') as f:
                self.le = pickle.load(f)
            self.labels = self.le.classes_
            self.estimator = get_estimator(
                self.model_checkpoint, self.bert_config_file, self.labels,
                self.predict_batch_size)

    def set_train_test_params(self, params):
        self.test = None
        train_params = params['train_params']
        self.init_checkpoint = train_params['init_checkpoint']
        self.model_dir = train_params['model_dir']
        self.train_batch_size = train_params['train_batch_size']
        self.learning_rate = train_params['learning_rate']
        self.num_train_epochs = train_params['num_train_epochs']
        self.warmup_proportion = train_params.get('warmup_proportion')
        self.save_checkpoints_steps = train_params.get('save_checkpoints_steps')
        self.save_summary_steps = train_params.get('save_summary_steps')
        train_params_data = train_params['data']
        try:
            train_folder = train_params_data['train_folder']
            print(f"Train folder: {train_folder}")
            test_folder = train_params_data.get('test_folder')
            print("[INFO] Loading train data from folder...")
            self.train, self.le = load_from_folder(train_folder)
            if test_folder:
                print("[INFO] Loading test data from folder...")
                self.test, _ = load_from_folder(test_folder, self.le)
        except:
            try:
                print("[INFO] Loading train data from csv...")
                train_csv = train_params_data['train_csv']
                test_csv = train_params_data.get('test_csv')
                self.train, self.le = load_dataset(train_csv)
                if test_csv:
                    print("[INFO] Loading test data from csv...")
                    self.test, _ = load_dataset(test_csv, self.le)
            except:
                raise Exception("None of the `train_folder` or `train_csv` is "
                                "mentioned in configuration")
        self.labels = self.le.classes_
        with open(self.label_encoder_pkl, 'wb') as f:
            pickle.dump(self.le, f)

    def do_train(self):
        print("[INFO] Preparing train InputExample...")
        train_inputExamples = self.train.apply(
            lambda x: run_classifier.InputExample(
                guid=None,
                text_a=x['sentence'],
                text_b=None,
                label=x['label']), axis=1)
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

        num_train_steps = \
            int(len(train_features)/self.train_batch_size*self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        print(f"[INFO] No. of train steps: {num_train_steps}")
        print(f"[INFO] No. of warmup steps: {num_warmup_steps}")

        self.estimator = get_estimator(
            self.init_checkpoint, self.bert_config_file, self.labels,
            self.train_batch_size, self.model_dir,
            save_summary_steps=self.save_summary_steps,
            save_checkpoints_steps=self.save_checkpoints_steps,
            learning_rate=self.learning_rate, num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        print(f'[INFO] Begin Training...!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print(f"[INFO] Training took time {datetime.now() - current_time} sec..!\n")

    def do_eval(self):
        print(f"[INFO] Started working on evaluation...")
        print("[INFO] Preparing test InputExample...")
        test_inputExamples = self.test.apply(
            lambda x: run_classifier.InputExample(
                guid=None,
                text_a=x['sentence'],
                text_b=None,
                label=x['label']), axis=1)
        print("[INFO] Done preparing test InputExample...\n")

        label_list = list(range(len(self.labels)))
        print("[INFO] Preparing test features...")
        test_features = run_classifier.convert_examples_to_features(
            test_inputExamples, label_list, self.max_seq_length, self.tokenizer)
        print("[INFO] Done preparing test features...\n")

        test_input_fn = run_classifier.input_fn_builder(
            features=test_features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        print(f'[INFO] Begin evaluating...!')
        result = self.estimator.evaluate(input_fn=test_input_fn, steps=None)
        print(f"[INFO] Done evaluating...\n")
        for key in sorted(result.keys()):
            print(f"[INFO]  {key} = {result[key]}")

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
        ret_predictions = []
        for pred in predictions:
            probabilities = pred[1]
            tops = np.argsort(probabilities)[::-1]
            top1_prob = probabilities[tops[0]]
            top2_prob = probabilities[tops[1]]
            second_top = self.labels[tops[1]]
            ret_predictions.append("TEXT: \"{}\" ==> {} {}".format(
                pred[0], pred[2], {pred[2]: top1_prob, second_top: top2_prob}))
            # for top in tops:
                # print(f"Label: {self.labels[top]} ==> {probabilities[top]}")
        for ret_prediction in ret_predictions:
            print(ret_prediction)
        return ret_predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', "--project", required=True)
    parser.add_argument("--cpu", action='store_true', help='Use CPU over GPU')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action='store_true')
    group.add_argument("--predict", action='store_true')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--file", help="Path to the file with each line as a "
                                       "record to be classified")
    group2.add_argument("-t", "--text", help="Enter the text to classify")
    group2.add_argument("--skip-eval", action='store_true')
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
    bc = BERTClassifier(project=args.project, mode=mode, skip_eval=args.skip_eval)
    if args.predict:
        bc.predict(args.file, args.text)
