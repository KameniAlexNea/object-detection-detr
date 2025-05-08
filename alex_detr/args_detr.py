from dataclasses import dataclass, field
from typing import List, Optional

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class ModelArgs:
    num_class: int = field(
        metadata={"help": "Number of classes supported"},
    )
    label_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of class names. If provided, its length must match num_class."},
    )
    training_csv: str = field(
        metadata={
            "help": "The output directory where data images and training csv are saved."
        },
        default="data/train_dataset.csv",
    )
    split_validation: float = field(
        metadata={"help": "validation split ration"}, default=None
    )
    image_folder: str = field(
        metadata={"help": "The output directory where data images are saved."},
        default="data/images/",
    )
    validation_csv: str = field(
        metadata={
            "help": "The output directory where data images and training csv are saved."
        },
        default=None,
    )
    # hustvl/yolos-small "jozhang97/deta-resnet-50-24-epochs" "hustvl/yolos-base" # "jozhang97/deta-resnet-50" # "facebook/detr-resnet-50"
    model_name: str = field(
        metadata={
            "help": "The output directory where data images and training csv are saved."
        },
        default="microsoft/conditional-detr-resnet-50",
    )

    nms_thr: float = field(
        metadata={"help": "NMS thr when compute metric"}, default=0.75
    )
    cls_thr: float = field(
        metadata={"help": "cls thr when compute metrics"}, default=0.25
    )
    nan_frac: float = field(
        metadata={"help": "proportion of images with empty labels to use in the training"},
        default=0.0
    )


@dataclass
class Argument:
    """
    Combines TrainingArguments from the transformers library with ModelArgs to provide a unified set of arguments.
    """

    training_args: TrainingArguments
    model_args: ModelArgs


def get_arguments():
    arguments = HfArgumentParser((TrainingArguments, ModelArgs))
    args = arguments.parse_args_into_dataclasses()
    args = Argument(*args)
    return args
