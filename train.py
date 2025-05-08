import os

os.environ["WANDB_PROJECT"] = "zindi_challenge_cacao"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "none"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    DetrForObjectDetection,
    EarlyStoppingCallback,
    Trainer,
)

from alex_detr.args_detr import get_arguments
from alex_detr.dataset import collate_fn, load_dataset, transform_aug_ann
from alex_detr.metrics import compute_metrics
from alex_detr.transforms import Config

args = get_arguments()


# use this for Detr with Focal Loss
# from alex_detr.model import DetrForObjectDetection


print("Start Training : ", os.getpid(), args.model_args.model_name)

label2id = {f"CL{i}": i for i in range(args.model_args.num_class)}
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
Config.NMS_THR = args.model_args.nms_thr
Config.IMAGE_FOLDER = args.model_args.image_folder
Config.TRAIN_CSV = args.model_args.training_csv


train_set = load_dataset(args.model_args.training_csv, nan_frac=args.model_args.nan_frac).with_transform(
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
    # compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)

if eval_set is not None:
    trainer.evaluate()
trainer.train()
if eval_set is not None:
    trainer.evaluate()
