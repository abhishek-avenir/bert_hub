
import tensorflow as tf
from datetime import datetime

from bert import run_classifier

from bert_hub import load_dataset, split_into_train_test, load_from_folder,\
	create_tokenizer_from_hub_module, model_fn_builder, getPrediction


# CSV_FILE = "data/data.csv"
train_folder = "imdb/train"
test_folder = "imdb/test"
LABELS = ['pos', 'neg']
DATA_COLUMN = "sentence"
LABEL_COLUMN = "label"

OUTPUT_DIR = "output_dir/"
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# dataset = load_dataset(CSV_FILE)
# train, test = split_into_train_test(dataset)
print("[INFO] Loading data from folders...")
train = load_from_folder(train_folder, labels=LABELS,
	data_column=DATA_COLUMN, label_column=LABEL_COLUMN)
test = load_from_folder(test_folder, labels=LABELS,
	data_column=DATA_COLUMN, label_column=LABEL_COLUMN)
print("[INFO] Done loading data...\n")

train = train.sample(5000)
test = test.sample(5000)

label_list = train[LABEL_COLUMN].unique()

print("[INFO] Preparing InputExample...")
train_inputExamples = train.apply(
	lambda x: run_classifier.InputExample(
		guid=None,
		text_a=x[DATA_COLUMN],
		text_b=None,
		label=x[LABEL_COLUMN]), axis=1)

test_inputExamples = test.apply(
	lambda x: run_classifier.InputExample(
		guid=None,
		text_a=x[DATA_COLUMN],
		text_b=None,
		label=x[LABEL_COLUMN]), axis=1)
print("[INFO] Done preparing InputExample...\n")

print("[INFO] Preparing Tokenizer...")
tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)
print("[INFO] Done preparing tokenizer...\n")

print("[INFO] Preparing features...")
train_features = run_classifier.convert_examples_to_features(
	train_inputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = run_classifier.convert_examples_to_features(
	test_inputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
print("[INFO] Done preparing features...\n")


num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
	bert_model_hub=BERT_MODEL_HUB,
	num_labels=len(label_list),
	learning_rate=LEARNING_RATE,
	num_train_steps=num_train_steps,
	num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
	model_fn=model_fn,
	config=run_config,
	params={"batch_size": BATCH_SIZE})

train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

print(f"[INFO] No. of train steps: {num_train_steps}")
print(f"[INFO] No. of warmup steps: {num_warmup_steps}")

print(f'[INFO] Beginning Training...!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print(f"[INFO] Training took time {datetime.now() - current_time} sec...!\n")

test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

print(f'[INFO] Beginning evaluating...!')
estimator.evaluate(input_fn=test_input_fn, steps=None)
print(f"[INFO] Done evaluating...\n")

pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!"
]
print(f"[INFO] Beginning predicting...")
predictions = getPrediction(
	estimator, pred_sentences, labels=LABELS, label_list=label_list,
	max_seq_len=MAX_SEQ_LENGTH, tokenizer=tokenizer)
print(f"[INFO] Done predicting...\n")
print(predictions)
