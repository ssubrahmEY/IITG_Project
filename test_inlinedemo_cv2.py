import cv2
import numpy as np

# Function to convert an image to grayscale
def convert_to_grayscale(image):
    if len(image.shape) == 3:  # If the image has 3 channels (color image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # If already grayscale or empty, return as is

# Test cases
def test_convert_to_grayscale():
    # Test case 1: Convert a color image to grayscale
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                     [[128, 128, 128], [64, 64, 64], [192, 192, 192]]], dtype=np.uint8)
    expected_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(convert_to_grayscale(image), expected_result)

    # Test case 2: Convert a grayscale image to grayscale
    grayscale_image = np.array([[0, 255, 255],
                               [128, 64, 192]], dtype=np.uint8)
    assert np.array_equal(convert_to_grayscale(grayscale_image), grayscale_image)

    # Test case 3: Convert an empty image
    empty_image = np.array([], dtype=np.uint8)
    assert np.array_equal(convert_to_grayscale(empty_image), empty_image)

    print("All tests passed!")

# Run the tests
test_convert_to_grayscale()