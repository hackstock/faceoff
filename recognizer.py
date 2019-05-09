import cv2
import os
import numpy as np 
from scipy import spatial

class FaceRecognizer(object):

    def __init__(self, encoder, db_dir, distance, haars_file, video_src=0):
        self.encoder = encoder
        self.db_dir = db_dir
        self.distance = distance
        self.classifier = cv2.CascadeClassifier(haars_file)
        self.video_cap = cv2.VideoCapture(video_src)

    def run(self):
        while(True):
            ret, frame = self.video_cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.putText(frame, "press 'r' to register a new user", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)

            faces = self.classifier.detectMultiScale(gray, 1.3, 5)
            roi = None
            for (x,y,w,h) in faces:
                roi = frame[y:y+h,x:x+w]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                if len(os.listdir(self.db_dir)) > 0:
                    self.match_face(roi, x, y, frame)

            cv2.imshow("Face Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.register_user(roi)

        self.video_cap.release()
        cv2.destroyAllWindows()

    def match_face(self, face, x, y, frame):
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        face = np.expand_dims(face, axis=0)
        encodings = self.encoder.predict(face)

        for known_face_file in os.listdir(self.db_dir):
            known_face_encodings = np.load("{}/{}".format(self.db_dir, known_face_file))
            similarity = spatial.distance.cosine(encodings, known_face_encodings)
            if similarity < self.distance:
                matched_username = known_face_file.replace(".npy", "").replace("_", " ")
                cv2.putText(frame, matched_username, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
            else:
                cv2.putText(frame, "unknown user", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

    def register_user(self,face):
        face = cv2.resize(face, (224,224), interpolation=cv2.INTER_AREA)
        face = np.expand_dims(face, axis=0)
        fullname = input("Enter your fullname : ")
        fullname = fullname.replace(" ", "_")
        encodings = self.encoder.predict(face)
        np.save("{}/{}".format(self.db_dir, fullname), encodings)
