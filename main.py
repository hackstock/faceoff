import argparse
import os

from model import build_face_encoder
from recognizer import FaceRecognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")

    parser.add_argument("--weights", dest="weights", required=True, help="path to pre-train vgg-face weights file")
    parser.add_argument("--dbdir", dest="dbdir", default="faces", help="path to directory for storing encoded faces")
    parser.add_argument("--distance", dest="distance", required=True, help="similarity threshold for determining matched faces")

    args = parser.parse_args()
    db_directory = args.dbdir 
    weights_file_path = args.weights
    distance = float(args.distance)

    if not os.path.exists(db_directory):
        os.mkdir(db_directory)

    haars_file = "./haarscascade.xml"

    face_encoder = build_face_encoder()
    face_encoder.load_weights(weights_file_path)

    face_recognizer = FaceRecognizer(encoder=face_encoder, db_dir=db_directory, distance=distance, haars_file=haars_file)
    face_recognizer.run()