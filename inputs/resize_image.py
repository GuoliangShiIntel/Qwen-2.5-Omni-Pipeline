from pathlib import Path
from PIL import Image

image_path = Path("C:/Intel/guoliang/omin_pipeline/inputs/user_behavior2.png")
image = Image.open(image_path)
print("Original size:", image.size)

def resize_image_use_factor(image, scale_factor):
    # Calculate the new size
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    new_size = (new_width, new_height)
    
    # Resize the image
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    if resized_image.mode == 'RGBA':
        resized_image = resized_image.convert('RGB')

    print("New size:", resized_image.size) 

    return resized_image

scale_factor = 0.5
resized_image = resize_image_use_factor(image, scale_factor)
image_resize_path = Path("C:/Intel/guoliang/omin_pipeline/inputs/user_behavior2_resize.png")
resized_image.save(image_resize_path)