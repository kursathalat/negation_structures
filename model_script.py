
# Sentiment analysis model on DistilBERT

## Install these HuggingFace libraries to run the model, if not installed
#pip install transformers evaluate datasets

import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
import datasets
import evaluate

corpus_to_upload = "train_corpus_first_epoch.npy"

loadTrainDictionary = np.load(corpus_to_upload,allow_pickle='TRUE').item()
loadTestDictionary = np.load('test_corpus.npy',allow_pickle='TRUE').item()

train = datasets.Dataset.from_dict(loadTrainDictionary)
test = datasets.Dataset.from_dict(loadTestDictionary)
corpus = datasets.DatasetDict({"train":train,"test":test})

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_corpus = corpus.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}



model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="negation_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_corpus["train"],
    eval_dataset=tokenized_corpus["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


## Uncomment to save the model

#trainer.save_model("negation_model")


## Quick use of the model

classifier = pipeline("sentiment-analysis", model="negation_model")
text = "I did not like this movie."
classifier(text)
