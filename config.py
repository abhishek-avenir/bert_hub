import os

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(ROOT_FOLDER, "data")

PATH_TO_MODELS = os.path.join(ROOT_FOLDER, "models")

PARAMS = {
	# Parameters for imdb project
	"imdb": {
		"vocab_file": "BERT_BASE_DIR/vocab.txt",
		"bert_config_file": "BERT_BASE_DIR/bert_config.json",
		"max_seq_length": 128,
		"label_encoder": os.path.join(PATH_TO_DATA, "bank", "label_encoder.pkl"),
		"labels": ['pos', 'neg'],
		"train_params": {
			"data": {
				"train_folder": os.path.join(PATH_TO_DATA, "imdb", "train"),
				"test_folder": os.path.join(PATH_TO_DATA, "imdb", "test"),
			},
			"init_checkpoint": "BERT_BASE_DIR/bert_model.ckpt",
			"model_dir": os.path.join(PATH_TO_MODELS, "imdb2"),
			"train_batch_size": 32,
			"learning_rate": 2e-5,
			"num_train_epochs": 3.0,
			"warmup_proportion": 0.1,
			"save_checkpoints_steps": 500,
			"save_summary_steps": 100,
		},
		"predict_params": {
			"predict_batch_size": 2,
			"model_checkpoint": os.path.join(PATH_TO_MODELS, "imdb", "model.ckpt-468"),
		},
	},


	"bank": {
		"vocab_file": "BERT_BASE_DIR/vocab.txt",
		"bert_config_file": "BERT_BASE_DIR/bert_config.json",
		"max_seq_length": 64,
		"label_encoder": os.path.join(PATH_TO_DATA, "bank", "label_encoder.pkl"),
		"train_params": {
			"data": {
				"train_csv": os.path.join(PATH_TO_DATA, "bank", "formatted_train.csv"),
				"test_csv": os.path.join(PATH_TO_DATA, "bank", "test.csv"),
			},
			"init_checkpoint": "BERT_BASE_DIR/bert_model.ckpt",
			"model_dir": os.path.join(PATH_TO_MODELS, "bank"),
			"train_batch_size": 16,
			"learning_rate": 2e-5,
			"num_train_epochs": 3.0,
			"warmup_proportion": 0.1,
			"save_checkpoints_steps": 50,
		},
		"predict_params": {
			"predict_batch_size": 10,
			"model_checkpoint": os.path.join(PATH_TO_MODELS, "bank", "model.ckpt-122"),
		},
	},
	"bank_gpu": {
		"vocab_file": "BERT_BASE_DIR/vocab.txt",
		"bert_config_file": "BERT_BASE_DIR/bert_config.json",
		"max_seq_length": 16,
		"label_encoder": os.path.join(PATH_TO_DATA, "bank", "label_encoder.pkl"),
		"train_params": {
			"data": {
				"train_csv": os.path.join(PATH_TO_DATA, "bank", "formatted_train.csv"),
				"test_csv": os.path.join(PATH_TO_DATA, "bank", "test.csv"),
			},
			"init_checkpoint": "BERT_BASE_DIR/bert_model.ckpt",
			"model_dir": os.path.join(PATH_TO_MODELS, "bank_gpu"),
			"train_batch_size": 1,
			"learning_rate": 2e-5,
			"num_train_epochs": 3.0,
			"warmup_proportion": 0.1,
			"save_checkpoints_steps": 50,
		},
		"predict_params": {
			"predict_batch_size": 10,
			"model_checkpoint": os.path.join(PATH_TO_MODELS, "bank_gpu", "model.ckpt-1956"),
		},
	},
}