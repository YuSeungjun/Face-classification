from __future__ import print_function
import face_recognition
import cv2
from sklearn.cluster import KMeans
import numpy as np
import os
import pickle
import signal
import sys


class Face():
    def __init__(self, frame_id, name, box, encoding):
        self.frame_id = frame_id
        self.name = name
        self.box = box
        self.encoding = encoding


class FaceClustering():
    def __init__(self):
        self.faces = []
        self.run_encoding = False
        self.capture_dir = "captures"

    def capture_filename(self, frame_id):
        return "frame_%08d.jpg" % frame_id

    def signal_handler(self, sig, frame):
        print(" stop encoding.")
        self.run_encoding = False

    def drawBoxes(self, frame, faces_in_frame):
        # Draw a box around the face
        for face in faces_in_frame:
            (top, right, bottom, left) = face.box
            width = right - left
            height = bottom - top
            left = max(0, left - int(width * 0.4))  # 얼굴 면적 40% 넓게 감지됨
            top = max(0, top - int(height * 0.4))  # 얼굴 면적 40% 넓게 감지됨
            right = min(frame.shape[1], right + int(width * 0.4))  # 얼굴 면적 40% 넓게 감지됨
            bottom = min(frame.shape[0], bottom + int(height * 0.4))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    def cluster_faces(self, n_clusters=12):
        encodings = [face.encoding for face in self.faces]
        if len(encodings) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
            labels = kmeans.fit_predict(encodings)

            for i, face in enumerate(self.faces):
                face.name = str(labels[i])  # assign cluster id as name

    def encode(self, src_file, capture_per_second, stop=0):
        src = cv2.VideoCapture(src_file)
        if not src.isOpened():
            return

        self.faces = []
        frame_id = 0
        frame_rate = src.get(5)
        stop_at_frame = int(stop * frame_rate)
        frames_between_capture = int(round(frame_rate) / capture_per_second)

        print("start encoding from src: %dx%d, %f frame/sec" % (src.get(3), src.get(4), frame_rate))
        print(" - capture every %d frame" % frames_between_capture)
        if stop_at_frame > 0:
            print(" - stop after %d frame" % stop_at_frame)

        # set SIGINT (^C) handler
        prev_handler = signal.signal(signal.SIGINT, self.signal_handler)
        print("press ^C to stop encoding immediately")

        if not os.path.exists(self.capture_dir):
            os.mkdir(self.capture_dir)

        individual_face_count = {}
        self.run_encoding = True
        while self.run_encoding:
            ret, frame = src.read()
            if frame is None:
                break

            frame_id += 1
            if frame_id % frames_between_capture != 0:
                continue

            if stop_at_frame > 0 and frame_id > stop_at_frame:
                break

            rgb = frame[:, :, ::-1]
            boxes = face_recognition.face_locations(rgb, model="hog")

            print("frame_id =", frame_id, boxes)
            if not boxes:
                continue

            encodings = face_recognition.face_encodings(rgb, boxes)

            faces_in_frame = []
            for box, encoding in zip(boxes, encodings):
                face = Face(frame_id, None, box, encoding)
                faces_in_frame.append(face)

            self.faces.extend(faces_in_frame)

            # Cluster the faces
            self.cluster_faces()

            # save the frame with drawn boxes
            self.drawBoxes(frame, faces_in_frame)
            
            # save individual faces
            for face in faces_in_frame:
                face_dir = os.path.join(self.capture_dir, f"cluster_{face.name}")
                if not os.path.exists(face_dir):
                    os.mkdir(face_dir)

                # Save box_frame in the same directory as individual faces
                box_frame_path = os.path.join(face_dir, f"frame_{frame_id:08d}_boxes.jpg")
                cv2.imwrite(box_frame_path, frame)

                face_path = os.path.join(face_dir, f"face_{frame_id:08d}.jpg")
                face_image = frame[face.box[0]:face.box[2], face.box[3]:face.box[1]]
                cv2.imwrite(face_path, face_image)

        # restore SIGINT (^C) handler
        signal.signal(signal.SIGINT, prev_handler)
        self.run_encoding = False
        src.release()
        return

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self.faces))

    def load(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
            self.faces = pickle.loads(data)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encode",
                    help="video file to encode or '0' to encode web cam")
    ap.add_argument("-c", "--capture", default=1, type=int,
                    help="# of frame to capture per second")
    ap.add_argument("-s", "--stop", default=0, type=int,
                    help="stop encoding after # seconds")
    args = ap.parse_args()

    fc = FaceClustering()

    if args.encode:
        src_file = args.encode
        if src_file == "0":
            src_file = 0
        fc.encode(src_file, args.capture, args.stop)
        fc.save("encodings.pickle")