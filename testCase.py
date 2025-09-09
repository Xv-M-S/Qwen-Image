

regional_prompt_mask_pairs = {
    "0": {
        "description": '''a chalkboard sign reading "Qwen Coffee 😊 $2 per cup"''',
        "mask": [128, 240, 384, 640]
    },
    "1": {
        "description": '''a neon light  displaying "通义千问"''',
        "mask": [500, 48, 840, 160]
    },
    "2": {
        "description": '''"π≈3.1415926-53589793-23846264-33832795-02384197" is written on the wall''',
        "mask": [500, 640, 756, 780]
    }
}

regional_prompt_mask_pairs2 = {
    "0": {
        "description": '''"咖啡店"''',
        "mask": [128, 240, 256, 640]
    },
    "1": {
        "description": '''"通义千问"''',
        "mask": [500, 48, 840, 160]
    },
    "2": {
        "description": '''"π≈3.1415926"''',
        "mask": [500, 640, 756, 780]
    }
}