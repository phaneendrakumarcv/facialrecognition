import face_recognition
import cv2
import os


known_faces = r"/home/phaneendra/Downloads/phaniimages/images"
unknown_faces = r"/home/phaneendra/Downloads/phaniimages/unknown"

known = []
known_names = []

for filename in os.listdir(known_faces):
    image = face_recognition.load_image_file(known_faces+"//"+filename)
    encoding = face_recognition.face_encodings(image)[0]
    known.append(encoding)
    known_names.append("Phaneendra")


print("Training on Unknown faces")

for filename in os.listdir(unknown_faces):
    image = face_recognition.load_image_file(unknown_faces+"//"+filename)
    locations = face_recognition.face_locations(image,model='cnn')
    encodings = face_recognition.face_encodings(image,locations)

    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    for face_encoding,face_location in zip(encodings,locations):
        results = face_recognition.compare_faces(known,face_encoding,0.6)
        if True in results:
            match = known_names[results.index(True)]
            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1],face_location[2])
            cv2.rectangle(image,top_left,bottom_right,(255,0,0),3)
            data_top_left = (face_location[3],face_location[0])
            data_top_right = (face_location[1],face_location[2]+24)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

cv2.imshow(filename,image)
cv2.waitKey(0)
cv2.destroyAllWindows()

