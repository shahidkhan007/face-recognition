from sql_handler import base64_to_array, split_base64
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


EYE_SIZE = 15

# Functionality: Performs face recognition to see if the user is in the databse or not
# How it works?: Creates a feed and then continually grabs a frame from it until a face
# is found, when found, it compares it to all the users's faces to see if it belongs to them,
# if yes, it returrns that user, otherwise it returns None, saying the face matching no one in the database.
def face_recognition(detector, f_model, all_users, draw_face_features=True, show_scan=True):
    try:
        # Get a reference to webcam #0 (the default one)
        feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Initial value
        face, box = None, None

        # recognition process
        while True:
            # Grab a single frame of video
            ret, frame = feed.read()

            # Exit if user presses 'q', required to get cv2 to work
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # if frame is retrieved successfully
            if ret:
                # search for a face in the frame
                face, features, is_face = detect_single_face(detector, frame)

                # Draw the face box and eye boxes if face was detected
                if draw_face_features and is_face:
                    frame = draw_features(frame, features)
                
                # whether to show a window to show the scan or not
                if show_scan:
                    cv2.imshow('Face recognition', frame)
                    
                    # Exits from the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Breaks the loop if face is detected to proceed to comparison
                if is_face:
                    break
        
        # Compare the user to all users in the database
        for user in all_users:
            # face encodings of the user in the database
            face_encodings = user.faces

            # Checks to see if this is the same person as the one in the database
            avg_dist, is_same_person, confidence, threshold = face_verification(f_model, face_encodings, face, threshold=1.1)
            if is_same_person:
                return user
            
        else:
            return None

    finally:
        feed.release()
        cv2.destroyAllWindows()


# Functionality: Detects multiple faces in a frame using the provided detector model
# How it works?: uses the pre-built MTCNN model to detect all the faces in the frame
# provided ( pixels ). The MTCNN models returns a list of dictionaries containing the faces detected,
# if any. the dictionary contains info like the bounding box, and face features like eyes, nose etc.
def detect_faces(detector, pixels):
    # Converting from cv2 BGR to RGB
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

    # Detector detects faces and returns a list of face data dictionaries for each face detected
    faces_data = detector.detect_faces(pixels)

    # Carve out the faces from the frame and the face features like the bounding box, eyes
    faces, features = extract_faces(pixels, faces_data)

    return faces, features


# Same as 'detect_faces' but this is for one face, I did this to avoid looping and save compute resources
def detect_single_face(detector, pixels):
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    faces_data = detector.detect_faces(pixels)
    face, features, is_face = extract_single_face(pixels, faces_data)

    return face, features, is_face



# Functionality: Carves out the faces from the frame and the face features like the bounding box, eyes
# How it works?: Uses the info dictionary provided by the MTCNN model to extract the face, as it will later be needed
# in face recognition, it needs a face, not the whole frame
def extract_faces(frame, frame_data):
    faces = []
    boxes = []

    # Face box coordinates
    coordinates = list( map( lambda result: result['box'], frame_data ) )

    for coord in coordinates:
        # Left top and right bottom coordinates of the face box
        x1, y1, width, height = coord
        x2, y2 = x1 + width, y1 + height

        # Constraining the dimensions of the box to be within the frame
        x1 = constrain(x1, 0, frame.shape[0])
        x2 = constrain(x2, 0, frame.shape[0])
        y1 = constrain(y1, 0, frame.shape[1])
        y2 = constrain(y2, 0, frame.shape[1])

        face = frame[y1 : y2, x1 : x2]
        faces.append(face)

        boxes.append( (x1, y1, x2, y2) )

    return faces, boxes


# Functionality: Draws the bounding box and eye features onto a frame
# How it works?: Uses the features disctionary provided and open-cv to draw the features on to the frame
def draw_features(frame, features):
    face = features['face']
    eye_left = features['eye_left']
    eye_right = features['eye_right']

    cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)
    cv2.circle(frame, eye_left, EYE_SIZE, (255, 0, 0), 2)
    cv2.circle(frame, eye_right, EYE_SIZE, (255, 0, 0), 2)

    return frame

# Same as 'extract_faces' but this is for one face, I did this to avoid looping and save compute resources
def extract_single_face(frame, frame_data):
    features = {}
    is_face = len(frame_data) > 0
    frame_width = frame.shape[0]
    frame_height = frame.shape[1]

    # empty frame_data means, no face detected
    if len(frame_data) == 0:
        return None, None, is_face
    
    # Extracting data for only a single face
    face_data = frame_data[0]

    x1, y1, width, height = face_data['box']
    x2, y2 = x1 + width, y1 + height

    # Constraining the dimensions of the box to be within the frame
    x1 = constrain(x1, 0, frame_width)
    x2 = constrain(x2, 0, frame_width)
    y1 = constrain(y1, 0, frame_height)
    y2 = constrain(y2, 0, frame_height)

    face = frame[x1: x2, y1: y2]

    # Saving all the features in a dict to return
    features['face'] = (x1, y1, x2, y2)
    features['eye_left'] = face_data['keypoints']['left_eye']
    features['eye_right'] = face_data['keypoints']['right_eye']


    return face, features, is_face



# Functionality: Uses the facenet model to convert a face to its encoding
# How it works?: first it converts the array to a PIL image, to resize it.
# Then, after its resized, its converted back to numpy array to add one more dimension to it
# as the model( Converts a carved face to a 512-dimensional vector ) expects multiple images.
# After the model outputs the the array, its dimensionality is decreased to what we want as
# the model outputs as to spit multiple encodings for multiple faces.
def img_to_encoding(facenet_model, face_array):
    face_image = Image.fromarray(face_array)
    face_image = face_image.resize((160, 160))  # resizing to match the model dimensionality
    face_array = np.asarray(face_image)

    face_array = np.expand_dims(face_array, 0)

    encoding = np.squeeze(facenet_model.embeddings(face_array))

    return encoding.astype(np.float32)


# Functionality: Finds and compares the distance between two faces using L2 norm
# How it works?: first we perform an element-wise subtraction and then take its L2 norm and
# this gives the distance between the 2 encodings, telling how much it thinks the encodings are of the same person
def compare_encodings(identity_encoding, face_encoding, threshold=3.0):
    sub = face_encoding - identity_encoding
    dist = np.linalg.norm(sub)

    if dist <= threshold:
        return True, dist
    else:
        return False, dist


# Functionality: Compares two faces and returns the average distance between them and if they are the same person based on the threshold parameter
# How it works?: Uses compare_encodings under the hood, it just does an ensemble type comparison for realiable distance.
def face_verification(model, other_user_encodings, face_array, threshold=3.0):
    # Convert the face to its encoding
    face_encoding = img_to_encoding(model, face_array)

    # faces saves in the DB
    faces = split_base64(other_user_encodings)
    faces = list( map( lambda face_b64: base64_to_array(face_b64), faces ) )

    # Distance of each user face to the face_array
    distances = []

    # number of times the face_array's encoding is closer to identity face than the threshold
    is_id_count = 0

    for enc in faces:
        is_face, dist = compare_encodings(enc, face_encoding)
        distances.append(dist)
        is_id_count += int(is_face)

    avg_dist = np.mean(distances)
    confidence = is_id_count / len(faces)

    return (avg_dist, avg_dist <= threshold, confidence, threshold)


# A function that contraints a value between a min and max
def constrain(value, min_value, max_value):  
    return min(max_value, max(min_value, value))


if __name__ == '__main__':
    print(constrain(10, 5, 15))
    print(constrain(10, 5, 7))
    print(constrain(10, 15, 20))