import cv2
import numpy as np
import face_recognition

imgStudent = face_recognition.load_image_file("ImagesBasic/Jahirul Haque Onik.jpg")
imgStudent = cv2.cvtColor(imgStudent, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("ImagesBasic/Rahat.PNG")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgStudent)[0]
encodeStudent = face_recognition.face_encodings(imgStudent)[0]
cv2.rectangle(imgStudent, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (225, 0, 225), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (225, 0, 22), 2)

results = face_recognition.compare_faces([encodeStudent], encodeTest)
faceDis = face_recognition.face_distance([encodeStudent], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results}{round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2)

cv2.imshow('Student', imgStudent)
cv2.imshow('Student Test', imgTest)
cv2.waitKey(0)