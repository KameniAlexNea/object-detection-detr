import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from alex_detr import IMAGE_FOLDER, TRAIN_CSV
from alex_detr.transforms import (
    Config
)


def _check_nan(category_id):
    return category_id is not None and not np.isnan(category_id)

def dropna_from_df(data: pd.DataFrame, frac: bool = 0.0):
    if not frac:
        return data.dropna()
    data_wna = data.dropna()
    data_na = data[data["bbox"].isna()]

    return pd.concat([data_wna, data_na.sample(random_state=41, frac=frac),])


def load_pd_dataframe(data_pth: str, training: bool = False, frac: bool = 0.0):
    train = pd.read_csv(data_pth)
    if training:  # drop empty images in training
        print(train.isna().sum())
        train = dropna_from_df(train, frac)
        print(train.isna().sum())
    train["image_id_int"] = LabelEncoder().fit_transform(train["image_id"])
    return train


def _create_bbox_data(bbox, category, id):
    bbox = json.loads(bbox)
    return {
        "id": id,
        "area": bbox[2] * bbox[3],
        "bbox": bbox,
        "category": int(category) - 1,  # convert 1,2,3 => 0,1,2
    }


def _to_dict(objects):
    return {
        "id": [i["id"] for i in objects if i["bbox"][2] and i["bbox"][3]],
        "area": [i["area"] for i in objects if i["bbox"][2] and i["bbox"][3]],
        "bbox": [i["bbox"] for i in objects if i["bbox"][2] and i["bbox"][3]],
        "category": [i["category"] for i in objects if i["bbox"][2] and i["bbox"][3]],
    }


def create_annotation_img(data: pd.DataFrame):
    image_id = data["image_id"].values[0]
    image_id_int = data["image_id_int"].values[0]
    objects = _to_dict(
        [
            _create_bbox_data(bbox, category_id, id)
            for (_, bbox, category_id, id, _) in data.values
            if _check_nan(category_id)
        ]
    )

    img = Image.open(os.path.join(IMAGE_FOLDER, f"{image_id}.tif"))
    width, height = img.size
    return {
        "image_id": image_id_int,
        "image": img,
        "width": width,
        "height": height,
        "objects": objects,
    }


def load_dataset(data_pth=TRAIN_CSV, training: bool = True, nan_frac: bool = 0.0):
    train = load_pd_dataframe(data_pth, training, nan_frac)
    train_features = (
        train.groupby("image_id")[train.columns].apply(create_annotation_img).tolist()
    )
    train_ds = Dataset.from_list(train_features)
    return train_ds


def _formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


# transforming a batch
def transform_aug_ann(examples, is_test=False):
    trf = Config.TRAIN_TRANSFORM if not is_test else Config.EVAL_TRANSFORM
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = trf(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": _formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return Config.IMAGE_PROCESSOR(images=images, annotations=targets, return_tensors="pt")


def train_val_split(train_ds: Dataset, test_size=0.1, seed=51):
    dataset = train_ds.train_test_split(test_size=test_size, seed=seed)
    train_set = dataset["train"].with_transform(transform_aug_ann)
    eval_set = dataset["test"].with_transform(lambda x: transform_aug_ann(x, True))
    return train_set, eval_set


def collate_fn(batch: list):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = Config.IMAGE_PROCESSOR.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    if "pixel_mask" in encoding:
        batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
