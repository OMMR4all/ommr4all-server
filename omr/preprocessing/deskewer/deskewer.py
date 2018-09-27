import numpy as np
from scipy.misc import imrotate
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from PIL import Image
import cv2

from omr.preprocessing.binarizer.ocropus_binarizer import binarize, normalize_raw_image, OCRopusBin


def extract_line_angles(staff_binary):
    lines = cv2.HoughLines(staff_binary, 1, np.pi / 180, 200)
    if lines is None:
        return []

    angles = []
    for (rho, theta), in lines:
        angle = theta * 180 / np.pi - 90
        if np.abs(angle) < 5:
            angles.append(angle)

    return angles


def gray_to_staff_binary(gray_image):
    binarized = 1 - binarize(normalize_raw_image(gray_image))
    morph = binary_erosion(binarized, structure=np.full((5, 1), 1))
    morph = binary_dilation(morph, structure=np.full((5, 1), 1))

    staffs = (binarized  ^ morph)
    return staffs.astype(np.uint8)


def estimate_rotation(gray_image: np.ndarray):
    angles_to_test = [-3, -2, -1, 0, 1, 2, 3]
    rotated_images = [imrotate(gray_image, angle) for angle in angles_to_test]
    rotated_images = map(gray_to_staff_binary, rotated_images)
    line_angles = map(extract_line_angles, rotated_images)
    all_angles = []
    for la, ang in zip(line_angles, angles_to_test):
        all_angles += [l + ang for l in la]

    return np.median(all_angles)


def deskew(original_image: Image, gray_image: Image):
    rot_angle = estimate_rotation(np.array(gray_image))
    o = original_image.rotate(rot_angle)
    g = gray_image.rotate(rot_angle)
    b = OCRopusBin().binarize(o)

    return o, g, b