from face import alignment
from face import detector
import argparse
import os
import cv2
from os.path import join

MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"


def __parse_args__():
    parser = argparse.ArgumentParser(description='Align all faces in a directory')
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--w', type=int)
    parser.add_argument('--h', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = __parse_args__()

    for folder in os.listdir(args.input_folder):

        if folder == ".DS_Store":
            continue

        print(f"Processing folder {folder}")
        in_im_folder = join(args.input_folder, folder)
        out_im_folder = join(args.output_folder, folder)

        if not os.path.exists(out_im_folder):
            os.mkdir(out_im_folder)

        for im_file_name in os.listdir(in_im_folder):
            if im_file_name == ".DS_Store":
                continue

            print(f"Aligning image {im_file_name}")
            im_file_path = join(in_im_folder, im_file_name)
            out_file_path = join(out_im_folder, im_file_name)

            im = cv2.imread(im_file_path, -1)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            landmarks = detector.predict_face_landmarks(im, MODEL_PATH)
            if len(landmarks) is not 1:
                print(f"Skipping image {im_file_name}")
                continue

            im_out, _ = alignment.normalize_images_landmarks(im, (args.h, args.w), landmarks[0])
            im_out.save(out_file_path)
            print(f"Aligning image {im_file_name} completed")
