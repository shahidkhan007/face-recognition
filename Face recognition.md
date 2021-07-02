# Algorithm

## 1. Models used

These are open-source models craeted by prestigous companies and are likely better than ones you would build on your own.

1. FaceNet model used for encoding a face into a 512-dimension vector.
2. MTCNN model used for detecting faces in images.

## 2. Workflow

1. A frame is captured from the webcam feed provided by cv2.
2. The frame is processed and the face is detected using MTCNN.
3. If there is a face detected, its carved out of the image using the information provided by MTCNN.
4. The face is encoded using the FaceNet model to a 512-dimension vector and this is the face encoding.
5. The encoding is compared with the encoding of the known faces from the database, which are grabbed at the beginning of the program.
6. If there is a match, the user is returned.
