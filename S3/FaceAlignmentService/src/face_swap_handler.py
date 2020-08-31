try:
    import unzip_requirements
except ImportError:
    pass
import io
import json
import base64
import cv2
from src.face import swapper
import numpy as np

from requests_toolbelt.multipart import decoder

MODEL_LOCAL_NAME = "models/shape_predictor_68_face_landmarks.dat"
H, W = 600, 600


def get_images(event):
    content_type_header = event['headers']['content-type']
    body = base64.b64decode(event['body'])
    print("Loaded body of the request")

    pic = decoder.MultipartDecoder(body, content_type_header).parts

    img1 = cv2.imdecode(np.frombuffer(pic[0].content, np.uint8), -1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

    img2 = cv2.imdecode(np.frombuffer(pic[1].content, np.uint8), -1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    return img1, img2


def swap_faces(event, context):
    """
    Swap face from img2 with face from img1.
    """
    try:

        im1, im2 = get_images(event)
        print("Loaded the image from input")

        im_out = swapper.swap(im1, im2, MODEL_LOCAL_NAME)

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
