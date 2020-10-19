from lib import dataset

if __name__ == "__main__":
    sp_t = dataset.SpatialTransformRepository().get_transform_obj("transforms.json")
    tp_t = dataset.TemporalTransformRepository().get_transform_obj("transforms.json")
    ds = dataset.VideoDataRepository(
        "/groups1/gaa50131/datasets/kinetics",
        "videos_700_hdf5",
        "kinetics-700-hdf5.json",
        spatial_transform=sp_t,
        temporal_transform=tp_t,
        clip_len=5,
        n_clip=8,
        downsample=3,
        mode=dataset.Mode.TRAIN,
    )
    for i in range(10):
        print(ds[i].clip.size())
        print(ds[i].label)
