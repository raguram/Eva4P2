try:
    import unzip_requirements
except ImportError:
    pass

import torch
import os
import io
import json
import base64
import boto3

from requests_toolbelt.multipart import decoder
from src.predictor import classifier

S3_Bucket = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'e4p2-models-ragu'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobileNetV2.pt'

print(f"Downloading model {MODEL_PATH}")
s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_Bucket, Key=MODEL_PATH)
        print("Creating ByteStream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading model")
        model = torch.jit.load(bytestream)
        print(f"Model loaded: {MODEL_PATH}")
except Exception as e:
    print(repr(e))
    raise (e)


def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event['body'])
        print("Loaded body of the request")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = classifier.classify(image_bytes=picture.content, model=model)
        print(f"Prediction: {prediction}")
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted': prediction})
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
