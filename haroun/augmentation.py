import numpy as np
import skimage.exposure as ex


def flip(images, labels, axis):
    flipped_images = np.flip(images, axis)
    flipped_labels = labels
    return flipped_images, flipped_labels


def brightness(images, labels, gamma):
    brightness_images = np.array([ex.adjust_gamma(image, gamma, gain=1) for image in images])
    brightness_labels = labels
    return brightness_images, brightness_labels


def augmentation(images, labels, flip_y, flip_x, brightness):
    if flip_y:
        # Data augmentation (flip_horizontal)
        flipped_y_images, flipped_y_labels = flip(images, labels, axis=2)

        # Concatenate arrays
        images = np.concatenate([images, flipped_y_images])
        labels = np.concatenate([labels, flipped_y_labels])

    if flip_x:
        # Data augmentation (flip_horizontal)
        flipped_x_images, flipped_x_labels = flip(images, labels, axis=1)

        # Concatenate arrays
        images = np.concatenate([images, flipped_x_images])
        labels = np.concatenate([labels, flipped_x_labels])

    if brightness:
        darken_images, darken_labels = brightness(images, labels, gamma=1.5)
        brighten_images, brighten_labels = brightness(images, labels, gamma=0.5)

        # Concatenate arrays
        images = np.concatenate([images, darken_images, brighten_images])
        labels = np.concatenate([labels, darken_labels, brighten_labels])

    return images, labels
