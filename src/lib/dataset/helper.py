from torch.utils.data import DataLoader

from . import utils
from . import video_dataset


def get_dataloader(CONFIG, finetune=False):
    if not finetune:
        tr_sp_t, tr_tp_t = utils.get_transforms(
            "train",
            resize=CONFIG.resize,
            clip_len=CONFIG.clip_len,
            n_clip=CONFIG.n_clip,
            downsample=CONFIG.downsample,
            consistent=CONFIG.model == "DPC",
        )
        val_sp_t, val_tp_t = utils.get_transforms(
            "val",
            resize=CONFIG.resize,
            clip_len=CONFIG.clip_len,
            n_clip=CONFIG.n_clip,
            downsample=CONFIG.downsample,
        )
    else:
        tr_sp_t, tr_tp_t = utils.get_transforms_finetune(
            "train",
            resize=CONFIG.resize,
            clip_len=CONFIG.clip_len,
            n_clip=CONFIG.n_clip,
            downsample=CONFIG.downsample,
        )
        val_sp_t, val_tp_t = utils.get_transforms_finetune(
            "val",
            resize=CONFIG.resize,
            clip_len=CONFIG.clip_len,
            n_clip=CONFIG.n_clip,
            downsample=CONFIG.downsample,
        )
    assert CONFIG.dataset in ("kinetics", "ucf", "hmdb")
    train_ds = video_dataset.VideoDataset(
        CONFIG.data_path,
        CONFIG.video_path,
        CONFIG.ann_path,
        clip_len=CONFIG.clip_len,
        n_clip=CONFIG.n_clip,
        downsample=CONFIG.downsample,
        spatial_transform=tr_sp_t,
        temporal_transform=tr_tp_t,
        mode="train",
    )
    val_ds = video_dataset.VideoDataset(
        CONFIG.data_path,
        CONFIG.video_path,
        CONFIG.ann_path,
        clip_len=CONFIG.clip_len,
        n_clip=CONFIG.n_clip,
        downsample=CONFIG.downsample,
        spatial_transform=val_sp_t,
        temporal_transform=val_tp_t,
        mode="val",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=utils.collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=utils.collate_fn,
    )
    return train_dl, val_dl
