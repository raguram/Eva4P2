import cv2
import numpy as np
from face import utils
from PIL import Image


def normalize_images_landmarks(im, out_size, landmark):
    points_in = np.array(utils.dlibLandmarksToPoints(landmark))
    im = np.float32(im) / 255.0

    h, w = out_size

    if (len(points_in)) == 68:
        eye_corner_src = [points_in[36], points_in[45]]
    elif len(points_in) == 5:
        eye_corner_src = [points_in[2], points_in[0]]
    else:
        raise Exception(f"Illegal argument. Length of points_in {len(points_in)}")

    eye_corner_dst = [(np.int(0.3 * w), np.int(h / 3)),
                      (np.int(0.7 * w), np.int(h / 3))]

    tform = utils.similarityTransform(eye_corner_src, eye_corner_dst)

    im_out = cv2.warpAffine(im, tform, (w, h))
    im_out = np.uint8(im_out * 255)

    points2 = np.reshape(points_in, (points_in.shape[0], 1, points_in.shape[1]))
    points_out = cv2.transform(points2, tform)

    points_out = np.reshape(points_out, (points_in.shape[0], points_in.shape[1]))
    return Image.fromarray(im_out), points_out
