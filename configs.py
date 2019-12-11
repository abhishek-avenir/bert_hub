import os

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(ROOT_FOLDER, "data")

PATH_TO_MODELS = os.path.join(ROOT_FOLDER, "models")

config = {
	"imdb": {
		"vocab_file": "BERT_BASE_DIR/vocab.txt",
		"bert_config_file": "BERT_BASE_DIR/bert_config.json",
		"max_seq_length": 128,
		"labels": ['pos', 'neg'],
		"model_dir": os.path.join(PATH_TO_MODELS, "imdb2"),
		"train_params": {
			"data": {
				"train_folder": os.path.join(PATH_TO_DATA, "imdb", "train"),
				"test_folder": os.path.join(PATH_TO_DATA, "imdb", "test"),
				"data_column": "sentence",
				"label_column": "label",
			},
			"init_checkpoint": "BERT_BASE_DIR/bert_model.ckpt",
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
	}
}