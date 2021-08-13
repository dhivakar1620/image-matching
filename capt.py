import cv2
import numpy as np
#kernel = np.ones((5,5), np.uint8)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
print(kernel)
dm=input("enter the disk name :")
cm=input("enter the cam side :")
#cm=1
cap = cv2.VideoCapture(1)
r=True
while(r):
    ret,frame = cap.read()
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = frame[50:250,200:500]#H,S
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.Canny(frame, 100, 200)
    cv2.imshow("view",frame)
    if (cv2.waitKey(1)&0xFF==ord('q')):
        break
print("done")
ss=str(dm+"-"+cm+".jpg")
cv2.imwrite(ss,frame)
cap.release()
cv2.destroyAllWindows()

