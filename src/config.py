architecture_config = [
    # tuple:(kernel size, # of filters of output, stride, padding)
    (7, 64, 2, 3),
    "M",  # max-pooling 2x2 stride = 2
    (3, 192, 1, 1),
    "M",  # max-pooling 2x2 stride = 2
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",  # max-pooling 2x2 stride = 2
    # [tuple, tuple, repeat times]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",  # max-pooling 2x2 stride = 2
    # [tuple, tuple, repeat times]
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]