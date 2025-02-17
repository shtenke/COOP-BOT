import cv2
import numpy as np

def process_image(image_path, output_path):
    numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') 
    plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    img = cv2.imread(image_path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plat_detector.detectMultiScale(img,scaleFactor=1.2,
        minNeighbors = 5, minSize=(25,25))   

    for (x,y,w,h) in plates:
        cv2.putText(img,text='License Plate',org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)
        img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],ksize=(100,100))
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imwrite(output_path,img)
    cv2.destroyAllWindows()
