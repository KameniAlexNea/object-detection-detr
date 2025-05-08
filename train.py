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
    TrainingArguments,
)

from alex_detr.args_detr import get_arguments, ModelArgs
from alex_detr.dataset import collate_fn, load_dataset, transform_aug_ann
from alex_detr.metrics import compute_metrics_fn
from alex_detr.transforms import Config as AugmentationConfig

args_tuple = get_arguments()
model_args: ModelArgs = args_tuple.model_args
training_args: TrainingArguments = args_tuple.training_args

print("Start Training : ", os.getpid(), model_args.model_name)

label2id = {f"CL{i}": i for i in range(model_args.num_class)}
id2label = {j: i for i, j in label2id.items()}

if model_args.names:
    if len(model_args.names) == model_args.num_class:
        label2id = {name: i for i, name in enumerate(model_args.names)}
        id2label = {i: name for i, name in enumerate(model_args.names)}
        print(f"Using custom label names: {model_args.names}")
    else:
        print(
            f"Warning: `label_names` provided but length ({len(model_args.names)}) "
            f"does not match `num_class` ({model_args.num_class}). Falling back to generic labels."
        )
        # Fallback to generic labels already handled by initial assignment

image_processor = AutoImageProcessor.from_pretrained(
    model_args.model_name,
)

model: DetrForObjectDetection = AutoModelForObjectDetection.from_pretrained(
    model_args.model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Prepare dataset transformations
train_transform = AugmentationConfig.TRAIN_TRANSFORM
eval_transform = AugmentationConfig.EVAL_TRANSFORM

# Wrapper for transform_aug_ann to pass necessary arguments
def _transform_aug_ann_train(examples):
    return transform_aug_ann(
        examples, image_processor, train_transform, eval_transform, is_test=False
    )

def _transform_aug_ann_eval(examples):
    return transform_aug_ann(
        examples, image_processor, train_transform, eval_transform, is_test=True
    )

train_set = load_dataset(
    model_args.training_csv, training=True, nan_frac=model_args.nan_frac
).with_transform(_transform_aug_ann_train)

eval_set = None
if model_args.validation_csv:
    eval_set = load_dataset(
        model_args.validation_csv, training=False
    ).with_transform(_transform_aug_ann_eval)
elif model_args.split_validation is not None:
    splitted_dt = train_set.train_test_split(test_size=model_args.split_validation)
    train_set, eval_set = splitted_dt["train"], splitted_dt["test"]
    eval_set = eval_set.with_transform(_transform_aug_ann_eval)

print("Dataset Shape", len(train_set), len(eval_set) if eval_set else "N/A")
if len(train_set) > 0:
    examples = train_set[0]
    print(examples)
else:
    print("Train set is empty.")

# Wrapper for collate_fn
def _collate_fn_wrapper(batch):
    return collate_fn(batch, image_processor)

# Prepare compute_metrics function
metrics_computer = compute_metrics_fn(num_classes=model_args.num_class)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=_collate_fn_wrapper,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=image_processor,
    # compute_metrics=metrics_computer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)

if eval_set is not None and training_args.do_eval:
    print("Evaluating before training...")
    trainer.evaluate()

if training_args.do_train:
    print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

if eval_set is not None and training_args.do_eval:
    print("Evaluating after training...")
    trainer.evaluate()
