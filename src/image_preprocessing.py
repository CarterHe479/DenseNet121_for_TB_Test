import numpy as np
from PIL import Image
from scipy.ndimage import rotate
from scipy.signal import correlate2d

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

# cross correlation analysis
def preprocess_template(template_img_path):
    template_img = Image.open(template_img_path)
    template_img_array = np.array(template_img)

    condition = (template_img_array[:, :, 0] > 115) & (template_img_array[:, :, 2] < 200)
    filtered_img = np.zeros_like(template_img_array)
    filtered_img[condition] = template_img_array[condition]

    template_gray = 0.6 * filtered_img[:, :, 0] + 0.3 * filtered_img[:, :, 1] + 0.1 * filtered_img[:, :, 2]
    return template_gray

def rotate_template(template_gray):
    return {
        'original': template_gray,
        '45': rotate(template_gray, 45, reshape=False),
        '90': rotate(template_gray, 90, reshape=False),
        '135': rotate(template_gray, -45, reshape=False)
    }
