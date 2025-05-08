import albumentations as A


class Config:
    TRAIN_TRANSFORM = A.Compose(
        [
            A.RandomScale(0.2, p=0.3),
            A.Rotate(limit=179, p=0.4),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(std_range=(0.002, 0.01), p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"], min_visibility=0.5
        ),
    )

    EVAL_TRANSFORM = A.Compose(
        []
    )
