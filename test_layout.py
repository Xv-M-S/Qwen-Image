from util.tool import draw_masks_on_image, visualize_mask_pairs, get_child_boxes
from util.textComposition import visualize_structured_boxes_with_text



if __name__ == "__main__":
    regional_prompt_mask_pairs = {
        "0": {
            "description": '''咖啡店"''',
            "mask": [128, 240, 256, 640]
        },
        "1": {
            "description": '''"通义千问"''',
            "mask": [500, 48, 840, 160]
        },
        "2": {
            "description": '''"π≈3.1415926"''',
            "mask": [500, 640, 756, 780]
        },
        # "3":{
        #     "description" : '''A poster showing a beautiful Chinese woman''',
        #     "mask": [900, 280, 1280, 680]
        # }
    }

    CANVAS_WIDTH = 1664
    CANVAS_HEIGHT = 928
   
    # 调用函数显示
    visualize_mask_pairs(regional_prompt_mask_pairs, CANVAS_WIDTH, CANVAS_HEIGHT)

    # image_path = "/home/sxm/flux-workspace/Qwen-Image/example.png"
    # 可视化在图片上
    # draw_masks_on_image(image_path=image_path, regional_prompt_mask_pairs=regional_prompt_mask_pairs)
    child_boxes = get_child_boxes(regional_prompt_mask_pairs, CANVAS_WIDTH, CANVAS_HEIGHT)
    print(child_boxes)

    visualize_structured_boxes_with_text(CANVAS_WIDTH, CANVAS_HEIGHT, child_boxes)