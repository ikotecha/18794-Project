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
    "loftr": {
        "output": "matches-loftr",
        "model": {"name": "loftr", "weights": "outdoor"},
        "preprocessing": {"grayscale": True, "resize_max": 1024, "dfactor": 8},
        "max_error": 1,  # max error for assigned keypoints (in px)
        "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
    },
    "superglue": {
        "output": "matches-superglue",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 50,
        },
    },
}