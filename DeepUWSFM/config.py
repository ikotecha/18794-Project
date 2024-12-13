# configs for different models
configs = {
    "superpoint": {
        "out_dir": "superpoint-feats-n4096-r1024",
        "model_stuff": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_kp": 4096,
        },
        "preprocess": {
            "gray": True,
            "max_size": 1024,
        },
    },
    "sift": {
        "output": "feats-sift",
        "model": {"name": "dog"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
}