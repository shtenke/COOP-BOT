import cv2
import numpy as np

def process_image(image_path, output_path):
    numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plat_detector.detectMultiScale(img,scaleFactor=1.2,
        minNeighbors = 5, minSize=(25,25))   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10,10))
    for (x,y,w,h) in plates:
        img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],ksize=(100,100))
    for (x, y, w, h) in faces:
        face_region = img[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        img[y:y+h, x:x+w] = blurred_face

    cv2.imwrite(output_path,img)
    cv2.destroyAllWindows()




