import pytesseract
from PIL import Image
import cv2
import os
import numpy as np
import face_recognition
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


config = { 'mask_color': (256, 256, 256), 
          'brut_psm': [6] }


input_file="D:/Projects/Aadhar_Masking/Input2.jpeg"
output="D:/Projects/Aadhar_Masking/Output Images/output2.jpg"


def box(img, psm):
        config  = ('-l eng --oem 3 --psm '+ str(psm))
        t = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT, config=config) 
        return t
    

def rotate(img):
        img_rotated = img   
        if is_image_upside_down(img_rotated):
            img_rotated_final = rotate_only(img_rotated)
  
            if is_image_upside_down(img_rotated_final):
        
                return img_rotated
            else:
                return img_rotated_final
        else:
            return img_rotated

def rotate_only(img):
        img = img
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
def is_image_upside_down(img):
        img = img
        face_locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, face_locations)
        image_is_upside_down = (len(encodings) == 0)
        return image_is_upside_down

aadhar_number=''
c=0
img = cv2.imread(input_file)


# cv2.imshow("Image",img)
# cv2.waitKey(0)
img=rotate(img)

# cv2.imshow("Image1",img)
# cv2.waitKey(0)
img2=img

#resize the image
#img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#convert the image to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 
text=pytesseract.image_to_string(img) 
res=text.split()           
       
       
for word in res: 
    if len(word) == 4 and word.isdigit():
        aadhar_number=aadhar_number  + word + ' '
        c=c+1
    if c==3:
        break
   
aadhar_number=aadhar_number.strip()
if len(aadhar_number)>=14:
    print("Aadhar number is :"+ aadhar_number)
else:
    print("Aadhar number not read")
    print("Try again or try  another file")
    
    
uid=aadhar_number.split(" ")
uid.remove(uid[2])

    
for i in range(len(config['brut_psm'])):      #'brut_psm': [6]
            d = box(img,config['brut_psm'][i])
            n_boxes = len(d['level'])
            color = config['mask_color']  #BGR
            for i in range(n_boxes):
                string = d['text'][i].strip()
                if string.isdigit() and string in uid and len(string)>=2:
                    #print('Number to be Masked =>',string)
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    #print("Rectangles =>",(x, y, w, h))
                    cv2.rectangle(img2, (x, y), (x + w, y + h), color, cv2.FILLED)
                    
                    

cv2.imwrite(output,img2)
cv2.imshow("output_file",img2)
cv2.waitKey(0)




    