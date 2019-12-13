
# Get Intent from the text

`config.py` contains the parameters required for the model training/evaluation and predicting.

Feel free to tune the parameters and build the model accordingly using the following command

### Training

Download and extract the zip contents (`bert_config.json`, `bert_model.ckpt` [all 3 files] and `vocab.txt`) of https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip to a new directory `BERT_BASE_DIR` under head directory (getIntent)

> python `bert_classify.py` --project bank --train


The model is generated under `models/bank` directory. Pick the top checkpoint and make sure "predict_params > model_checkpoint" parameter in the config.py is pointing to the path to the top checkpoint (or any other checkpoint you might want to use)

### Evaluate

> python `bert_classify.py` --project bank --predict --text "I want to transfer money to my another account"

or you can also send the text file (or any readable file) in which each line is an individual line for which you want to gather the intent

> python `bert_classify.py` --project bank --predict --file test.txt

