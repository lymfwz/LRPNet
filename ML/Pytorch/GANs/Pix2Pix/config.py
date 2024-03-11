import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/myAnime/myTrain"
VAL_DIR = "data/myAnime/myVal"
# TRAIN_DIR = "data/data/train"
# VAL_DIR = "data/data/val"
# TRAIN_DIR = "data/data/train"
# VAL_DIR = "data/data/val"
LEARNING_RATE = 2e-4 # ori : 2e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 10000
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
# CHECKPOINT_DISC = "pretrain/color/disc_300.pth.tar"
# CHECKPOINT_GEN = "process/color/gen_300.pth.tar"
# CHECKPOINT_DISC = "pretrain/line/disc_100.pth.tar"
# CHECKPOINT_GEN = "pretrain/line/gen_100.pth.tar"

both_transform = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        # 上色 #####################
        # A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        ###########################
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

def my_transform(x):
    return A.Compose(
        [
            A.Resize(width=x, height=x),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
            ToTensorV2(),
        ]
    )
