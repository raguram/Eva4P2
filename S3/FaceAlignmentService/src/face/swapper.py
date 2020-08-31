from src.face import utils
from src.face import detector
import numpy as np
import cv2
from PIL import Image


def swap(img1, img2, predictor_path):
    img1Warped = np.copy(img2)

    points1 = utils.dlibLandmarksToPoints(detector.predict_face_landmarks(img1, predictor_path)[0])
    points2 = utils.dlibLandmarksToPoints(detector.predict_face_landmarks(img2, predictor_path)[0])

    hullIndex = cv2.convexHull(points2, returnPoints=False)

    hull1 = []
    hull2 = []
    for i in range(0, len(hullIndex)):
        hull1.append(points1[hullIndex[i][0]])
        hull2.append(points2[hullIndex[i][0]])

    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    m = cv2.moments(mask[:, :, 1])
    center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = utils.calculateDelaunayTriangles(rect, hull2)

    tris1 = []
    tris2 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri2 = []
        for j in range(0, 3):
            tri1.append(hull1[dt[i][j]])
            tri2.append(hull2[dt[i][j]])

        tris1.append(tri1)
        tris2.append(tri2)

    for i in range(0, len(tris1)):
        utils.warpTriangle(img1, img1Warped, tris1[i], tris2[i])

    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    return Image.fromarray(output)
