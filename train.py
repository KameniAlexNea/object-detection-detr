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
    Trainer,
    AutoImageProcessor,
)

from alex_detr.dataset import (
    collate_fn,
    load_dataset,
    transform_aug_ann,
)
from alex_detr.transforms import Config
from alex_detr.metrics import compute_metrics

from alex_detr.args_detr import get_arguments

args = get_arguments()


# use this for Detr with Focal Loss
# from alex_detr.model import DetrForObjectDetection


print("Start Training : ", os.getpid(), args.model_args.model_name)

label2id = {"Thatch": 2, "Tin": 1, "Other": 0}
id2label = {j: i for i, j in label2id.items()}
model: DetrForObjectDetection = AutoModelForObjectDetection.from_pretrained(
    args.model_args.model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    # revision="main",
)
Config.IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(
    args.model_args.model_name,
)
Config.NUM_CLASS = args.model_args.num_class


train_set = load_dataset(args.model_args.training_csv, nan_frac=0.15).with_transform(
    transform_aug_ann
)
eval_set = None
if args.model_args.validation_csv:
    eval_set = load_dataset(args.model_args.validation_csv, False).with_transform(
        lambda x: transform_aug_ann(x, True)
    )
elif args.model_args.split_validation is not None:
    splitted_dt = train_set.train_test_split(test_size=args.model_args.split_validation)
    train_set, eval_set = splitted_dt["train"], splitted_dt["test"]

print("Dataset Shape", len(train_set), len(eval_set))
examples = train_set[0]
print(examples)

trainer = Trainer(
    model=model,
    args=args.training_args,
    data_collator=collate_fn,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=Config.IMAGE_PROCESSOR,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)
trainer.evaluate()
trainer.train()
trainer.evaluate()
