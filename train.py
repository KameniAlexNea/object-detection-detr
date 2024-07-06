import os


os.environ["WANDB_PROJECT"] = "zindi_challenge"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_NOTEBOOK_NAME"] = "train_hf"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    AutoModelForObjectDetection,
    DetrForObjectDetection,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from alex_detr import MODEL_NAME, TRAIN_CSV, VAL_CSV
from alex_detr.dataset import (
    collate_fn,
    load_dataset,
    transform_aug_ann,
)
from alex_detr.transforms import IMAGE_PROCESSOR
from alex_detr.metrics import compute_metrics


# from alex_detr.model import DetrForObjectDetection


print("Start Training : ", os.getpid(), MODEL_NAME)

label2id = {"Thatch": 2, "Tin": 1, "Other": 0}
id2label = {j: i for i, j in label2id.items()}
model: DetrForObjectDetection = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    # revision="main",
)


(training_args,) = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()

train_set = load_dataset(TRAIN_CSV, nan_frac=0.15).with_transform(transform_aug_ann)
eval_set = load_dataset(VAL_CSV, False).with_transform(
    lambda x: transform_aug_ann(x, True)
)
print("Dataset Shape", len(train_set), len(eval_set))
examples = train_set[0]
print(examples)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=IMAGE_PROCESSOR,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)
trainer.evaluate()
trainer.train()
trainer.evaluate()
