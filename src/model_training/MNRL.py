import pandas as pd
from datasets import IterableDataset, Features, Value
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from helpers.dataset_generator import balanced_generator, df_to_corpus
from helpers.InformationRetrievalEval import setup_evaluator
import os
import datetime 
import argparse
import mlflow
from transformers.integrations import MLflowCallback

#=================================================================================================
# Requires transformers==4.57.6 encountered issues, when using transformers>=5.XX.X
# Loss did't descrease properly
#=================================================================================================

#=================================================================================================
# Logging results
#=================================================================================================

# The results and model can be logged to mlflow mlflow to keep track of your training/versionise your models
# This class logs all the sentence transformers callbacks to mlflow. You'll receive visualisations of loss decay
# and the measured success metrics
# Requirement for that is that you start mlflow before you'll starte the training

class LoggingCallbacks(MLflowCallback):
    """
    Patched MlflowCallback, because we need to remove @ and turn it into _at_ because the 
    special character @ we receive when measuring recall@10 for example causes problems in mlflow
    Loggs all callbacks from the sentence transformers trainer
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Removing @ from the keys, due to issues with this certain character
        if logs:
            logs = {k.replace("@", "_at_"): v for k, v in logs.items()}
        return super().on_log(args, state, control, logs, **kwargs)


mlflow.set_tracking_uri(
  os.getenv("ML_FLOW_URI", "http://127.0.0.1:5000")
)

mlflow.set_experiment(
  os.getenv("MODEL_FINETUNING_EXPERIMENT", "Retrieval_Model_Training")
)

#=================================================================================================
# Loading, parsing and setting required variables.
#=================================================================================================

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--path_training_data_raw', type=str)      # option that takes a value
parser.add_argument('-s', '--path_training_data_storage', type=str)  
parser.add_argument('-o', '--output_dir', type=str)
parser.add_argument('-m', '--model_path', type=str)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-tc', '--text_column', type=str)
parser.add_argument('-lc', '--label_column', type=str)

args = parser.parse_args()



PATH_TRAINING_DATA_RAW = args.path_training_data_raw # Path where we load the raw training data from
PATH_TRAINING_DATA_STORAGE = args.path_training_data_storage # Path where we store the training data
OUTPUT_DIR = args.output_dir
MODEL_PATH = args.model_path
BATCH_SIZE= args.batch_size
TEXT_COLUMN = args.text_column
LABEL_COLUMN = args.label_column


mlflow.log_param("PATH_TRAINING_DATA_RAW", PATH_TRAINING_DATA_RAW) # Path where the full training data is stored
mlflow.log_param("PATH_TRAINING_DATA_STORAGE", PATH_TRAINING_DATA_STORAGE) # Path where to store store the training/test-dataset after train-test-split
mlflow.log_param("MODEL_PATH", MODEL_PATH) # If you're using a local model, specify the path to your model file here otherwise, just specify the hf-model id
mlflow.log_param("OUTPUT_DIR", OUTPUT_DIR) # Path where you want to store the trained model/checkpoints
mlflow.log_param("BATCH_SIZE", BATCH_SIZE) # Batch size for model training
# Assuming you store your training data in a tabular format, e.g. csv, parquet, ... 
mlflow.log_param("TEXT_COLUMN", TEXT_COLUMN) # Column name of the text/input you want to process
mlflow.log_param("LABEL_COLUMN", LABEL_COLUMN) # Column name where you store the labels

#=================================================================================================
# Loading Data and Constructing Corpus/Training/Testing Data
#=================================================================================================

df = pd.read_parquet(PATH_TRAINING_DATA_RAW)

train_df = df.sample(frac=0.9, random_state=42)
train_df.to_parquet(f'{PATH_TRAINING_DATA_STORAGE}/train_{str(datetime.datetime.now())[:10]}.parquet')

test_df = df.drop(train_df.index)
test_df.to_parquet(f'{PATH_TRAINING_DATA_STORAGE}/test_{str(datetime.datetime.now())[:10]}.parquet')



#=================================================================================================
# Creating train-datasets for the sentence-transformer-trainer
#=================================================================================================

unique_labels_train, corpus_train = df_to_corpus(
    df=train_df,
    text_column=TEXT_COLUMN,
    label_column=LABEL_COLUMN
)



features = Features({
    "anchor": Value("string"),
    "positive": Value("string")
})


def get_generator():
    yield from balanced_generator(BATCH_SIZE, corpus_train, unique_labels_train)

train_dataset = IterableDataset.from_generator(get_generator, features=features)
 
#=================================================================================================
# Loading model and definining loss, evaluator 
#=================================================================================================

model = SentenceTransformer(MODEL_PATH)
evaluator = setup_evaluator(
    test_df,
    text_column = TEXT_COLUMN,
    label_column = LABEL_COLUMN
)
loss = losses.MultipleNegativesRankingLoss(model)

#=================================================================================================
# training Model
#=================================================================================================


args = SentenceTransformerTrainingArguments(

    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    max_steps=7500,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=2e-5,
    logging_steps=100,
    eval_strategy="steps", 
    eval_steps=2500,
    disable_tqdm=True,
    report_to="none"
)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    args=args,
    loss=loss,
    evaluator=evaluator,
    callbacks=[LoggingCallbacks()]
)

trainer.train()

 

