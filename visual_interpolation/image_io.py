import cv2
import numpy as np


def image_as_tensor():
    """Get the original image as a ndarray[h, w, d]"""
    return cv2.imread("/Users/ericliao/PycharmProjects/phd_class_code/visual_interpolation/data/input2.webp")


def image_as_framelike():
    """Returns the input image in a dataframe style

    Returns (original_shape, indices as ndarray[h*w, 2], values as [h*w, d])
    """
    tensor = image_as_tensor()
    height, width, depth = tensor.shape
    y_indices = np.repeat(np.arange(height, dtype=np.uint16), width).reshape((-1, 1))
    x_indices = np.tile(np.arange(width, dtype=np.uint16), height).reshape((-1, 1))
    indices = np.concatenate([x_indices, y_indices], axis=1)

    return tensor.shape, indices, tensor.reshape((height * width, depth))

def image_to_train_test_all():
    """ Load the image like a frame, and split the data into three sets

        This uses train - test - all because it's more convenient to run the
        image on all the data to generate a new image

        Returns: ((train_xy, train_rgb), (test_xy, test_rgb), (original_shape, all_xy, all_rgb)) where:
            train_*: 100000 samples of the image
            test_*: A different 100000 samples of the image
            all_*: The whole image, 36 million+ samples
            *_xy: indices as a (*, 2) array
            *_rgb: colors as a (*, 3) array
    """
    shape, indices, values = image_as_framelike()
    gen = np.random.default_rng(3_1415926)
    both_rows = gen.choice(values.shape[0], 200000, replace=False)
    train_rows, test_rows = both_rows[:100000], both_rows[100000:]
    return (
        (indices[train_rows], values[train_rows]),
        (indices[test_rows], values[test_rows]),
        (shape, indices, values)
    )






def predited_framelike_to_image(
    original_shape, framelike, save_as: str = None, show: bool = False
):
    image = framelike.reshape(original_shape).clip(0, 255).astype(np.uint8)
    if save_as:
        cv2.imwrite(save_as, image)
    if show:
        cv2.imshow("Predicted Image", image)
        cv2.waitKey(5000)
    return image
