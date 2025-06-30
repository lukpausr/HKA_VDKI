
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import torch
from torchvision import transforms

from PIL import Image

import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Helper function to get bounding boxes for rabbits
def get_rabbit_boxes(image, threshold=0.4):                                                         #TODO: adjust threshold as necessary
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(image)
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    boxes = []
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()

    accepted_labels = [16, 17, 18, 19, 20, 21, 22, 23, 24, 88, 74]                                  # COCO classes for animals TODO: remove 16 (birds??)
    for box, label, score in zip(prediction['boxes'], labels, scores):
        if label in accepted_labels and score > threshold:
            relative_y_center = (box[1] + box[3]) / 2 / image.height
            relative_x_center = (box[0] + box[2]) / 2 / image.width
            add_box = True
            for boxbox in boxes:
                relative_x_center_boxbox = (boxbox[0] + boxbox[2]) / 2 / image.width
                relative_y_center_boxbox = (boxbox[1] + boxbox[3]) / 2 / image.height
                if abs(relative_x_center - relative_x_center_boxbox) < 0.1 and abs(relative_y_center - relative_y_center_boxbox) < 0.1:
                    # If the center of the new box is too close to an existing box, skip it
                    add_box = False
                    break
            if add_box: boxes.append(box.int().tolist())
    return boxes

def save_rabbit_crops(image_path, target_path=None, border=10):
    image = Image.open(image_path).convert("RGB")
    images = []
    boxes = get_rabbit_boxes(image)
    base, ext = os.path.splitext(os.path.basename(image_path))
    for idx, box in enumerate(boxes):
        image_copy=image.copy()  # Create a copy of the image to avoid modifying the original
        x1, y1, x2, y2 = box
        x1b = x1 - border
        y1b = y1 - border
        x2b = x2 + border
        y2b = y2 + border
        # Make crop quadratic
        width = x2b - x1b
        height = y2b - y1b
        side = max(width, height)
        center_x = (x1b + x2b) // 2
        center_y = (y1b + y2b) // 2
        half_side = side // 2
        x1q = center_x - half_side
        y1q = center_y - half_side
        x2q = center_x + half_side
        y2q = center_y + half_side
        if (x1q < 0 or y1q < 0 or x2q > image.width or y2q > image.height):
            # If the crop goes out of bounds, pad the image
            pad_l = max(0, -x1q)
            pad_t = max(0, -y1q)
            pad_r = max(0, x1q + side - image.width)
            pad_b = max(0, y1q + side - image.height)
            image_copy = transforms.functional.pad(image, (pad_l, pad_t, pad_r, pad_b), padding_mode="reflect")  # Pad the image if the crop goes out of bounds
            x1q = x1q + pad_l
            y1q = y1q + pad_t
            x2q = x2q + pad_l
            y2q = y2q + pad_t
            # print("picture transformed")
        
        image_copy = image_copy.crop((x1q, y1q, x2q, y2q))        
        if target_path is None:
            images.append(image_copy)
            continue
        else:
            image_copy.save(os.path.join(target_path, f"{base}_crop_{idx}{ext}"))
        image_copy.close()
    
    if len(boxes) == 0:
        print(f"No rabbit-like objects found in {base}. No crops available.")
        images.append(image)
    # image.close()
    return images if target_path is None else None

def get_animals(image, threshold=0.0):  #TODO: adjust threshold as necessary
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(image)
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    labels = prediction['labels'].tolist()
    scores = prediction['scores'].tolist()
    i = 0
    keep_indices = []
    for score in scores:
        if score > threshold:
            keep_indices.append(i)
        i += 1
    labels = [labels[i] for i in keep_indices]
    scores = [scores[i] for i in keep_indices]
    return labels, scores