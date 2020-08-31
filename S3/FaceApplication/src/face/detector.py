import dlib


def detect(im):
    face_detector = dlib.get_frontal_face_detector()
    face_rects = face_detector(im, 0)
    print("Number of faces detected: ", len(face_rects))
    return face_rects


def predict_face_landmarks(im, predictor_path, face_rects=None):
    if face_rects is None:
        face_rects = detect(im)

    landmark_detector = dlib.shape_predictor(predictor_path)
    all_landmarks = []
    for rect in face_rects:
        dlib_rect = dlib.rectangle(int(rect.left()),
                                   int(rect.top()),
                                   int(rect.right()),
                                   int(rect.bottom()))
        landmarks = landmark_detector(im, dlib_rect)
        print("Number of landmarks", len(landmarks.parts()))

        all_landmarks.append(landmarks)
    return all_landmarks
