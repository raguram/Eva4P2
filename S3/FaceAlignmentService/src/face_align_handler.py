try:
    import unzip_requirements
except ImportError:
    pass
import io
import json
import base64
import cv2
from src.face import detector
from src.face import alignment
import numpy as np

from requests_toolbelt.multipart import decoder

MODEL_LOCAL_NAME = "models/shape_predictor_5_face_landmarks.dat"
H, W = 600, 600


def get_image(event):
    content_type_header = event['headers']['content-type']
    body = base64.b64decode(event['body'])
    print("Loaded body of the request")

    pic = decoder.MultipartDecoder(body, content_type_header).parts[0]
    img = cv2.imdecode(np.frombuffer(pic.content, np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def align_face(event, context):
    """
    Takes in an image with one or more faces. Aligns each of the faces and returns a list of images with aligned faces.
    """
    try:

        im = get_image(event)
        print("Loaded the image from input")

        all_face_landmarks = detector.predict_face_landmarks(im, MODEL_LOCAL_NAME)

        if len(all_face_landmarks) > 1:
            return fail(f"{len(all_face_landmarks)} faces detected. Send image with only one face to align")
        if len(all_face_landmarks) == 0:
            return fail("No face detected. Send image with one face to align")

        im_out, points_out = alignment.normalize_images_landmarks(im, (H, W), all_face_landmarks[0])

        buffer = io.BytesIO()
        im_out.save(buffer, format="JPEG")
        output_bytes = base64.b64encode(buffer.getvalue())

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "image/jpeg",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": output_bytes
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }


def fail(message):
    return {
        "statusCode": 400,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": True
        },
        "body": message
    }
