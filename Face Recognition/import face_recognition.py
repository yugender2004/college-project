import face_recognition
import cv2
import numpy as np
import dlib
import csv
import os
from datetime import datetime

new_directory = "D:/Face Recognition"
os.chdir(new_directory)


anu_image = cv2.imread("img/anu.jpg")
yugender_image = cv2.imread("img/yugender.jpg")
akhila_image = cv2.imread("img/akhila.jpg")
surya_image = cv2.imread("img/surya.jpg")
tanya_image = cv2.imread("img/tanya.jpg")

anu_encoding = face_recognition.face_encodings(anu_image)[0]
yugender_encoding = face_recognition.face_encodings(yugender_image)[0]
akhila_encoding = face_recognition.face_encodings(akhila_image)[0]
surya_encoding = face_recognition.face_encodings(surya_image)[0]
tanya_encoding = face_recognition.face_encodings(tanya_image)[0]

known_face_encoding = [
    anu_encoding,
    yugender_encoding,
    akhila_encoding,
    surya_encoding,
    tanya_encoding
]
known_faces_names = [
    "Anu",
    "Yugender",
    "Akhila",
    "Surya",
    "Tanya"
]

students = known_faces_names.copy()

video_capture = cv2.VideoCapture(0)

now = datetime.now()
current_data = now.strftime("%Y-%m-%d")


csv_file_path = current_data + '_attendance.csv'


if not os.path.isfile(csv_file_path):
    with open(csv_file_path, 'w', newline='') as f:
        lnwriter = csv.writer(f)
        lnwriter.writerow(["Name", "Time"])

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding in face_encodings:
      
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_faces_names[first_match_index]
            print(f"Recognized: {name}")

        if name in known_faces_names:
            if name in students:
                students.remove(name)
                current_date  = now.strftime("%d/%m/%Y")
                current_time = now.strftime("%H:%M")
                dept = "CSBS"
                with open(csv_file_path, 'a', newline='') as f:
                    lnwriter = csv.writer(f)
                    lnwriter.writerow([name, dept, current_date, current_time])

    cv2.imshow("attendance system", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()
