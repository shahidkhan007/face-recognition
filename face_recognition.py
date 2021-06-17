from sql_handler import base64_to_array, split_base64
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image



def face_recognition(detector, f_model, all_users):
    try:
        feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        face = None

        while True:
            ret, frame = feed.read()
            if ret:
                faces = detect_faces(detector, frame)
                if len(faces) > 0:
                    face = faces[0]
                    break
        
        for user in all_users:
            face_encodings = user.faces
            avg_dist, is_same_person, confidence, threshold = face_verification(f_model, face_encodings, face, threshold=1.1)
            if is_same_person:
                return user
            
        else:
            return None

    finally:
        feed.release()
        cv2.destroyAllWindows()


def detect_faces(detector, pixels):
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    faces_data = detector.detect_faces(pixels)
    faces = extract_faces(pixels, faces_data)

    return faces


def extract_faces(frame, frame_data):
    faces = []
    coordinates = list( map( lambda result: result['box'], frame_data ) )
    for coord in coordinates:
        x1, y1, width, height = coord
        x2, y2 = x1 + width, y1 + height

        face = frame[y1 : y2, x1 : x2]
        faces.append(face)

    return faces


def img_to_encoding(facenet_model, face_array):
    face_image = Image.fromarray(face_array)
    face_image = face_image.resize((160, 160))
    face_array = np.asarray(face_image)

    face_array = np.expand_dims(face_array, 0)

    encoding = np.squeeze(facenet_model.embeddings(face_array))

    return encoding.astype(np.float32)


def compare_encodings(identity_encoding, face_encoding, threshold=3.0):
    sub = face_encoding - identity_encoding
    dist = np.linalg.norm(sub)

    if dist <= threshold:
        return True, dist
    else:
        return False, dist


def face_verification(model, other_user_encodings, face_array, threshold=3.0):
    face_encoding = img_to_encoding(model, face_array)

    faces = split_base64(other_user_encodings)
    faces = list( map( lambda face_b64: base64_to_array(face_b64), faces ) )

    distances = []
    is_id_count = 0

    for enc in faces:
        is_face, dist = compare_encodings(enc, face_encoding)
        distances.append(dist)
        is_id_count += int(is_face)

    avg_dist = np.mean(distances)
    confidence = is_id_count / len(faces)

    return (avg_dist, avg_dist <= threshold, confidence, threshold)
