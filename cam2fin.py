import cv2
import datetime 
import numpy as np
import winsound




thrmn={'hci13':110,'b90002':12,'bc3_2':5,'bc3_1':9,'0U3001':14,'b90001':8,'0U3002':14,'b90002':15,'ihhi':10,'c802':5,'c801':5,'GS1':1,'topp':17}#min thra
thrmx={'hci13':110,'b90002':12,'bc3_2':5,'bc3_1':9,'0U3001':14,'b90001':8,'0U3002':14,'b90002':15,'ihhi':10,'c802':5,'c801':5,'GS1':1,'topp':40}#max tha val
orb=cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)#points col
na='topp'
#naa='GS2'
imge1=cv2.imread(na+'.jpg',0)
#imge5=cv2.imread(naa+'.jpg',0)
lee=[]
le2=[]
th1=thrmn[na]
th2=thrmx[na]
fl=0
f2=0
sh=''
kernel = np.ones((5,5), np.uint8)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#converter = pyttsx3.init()   
#converter.setProperty('rate', 100) 
#converter.setProperty('volume', 1)

yu=0


def info(img):
    kp,des=orb.detectAndCompute(img,None)
    return(kp,des)

def matc(ig1,k1,d1,ig2,k2,d2,le,flg,thn,thx,nam,yy):
    good=[]
    bf=cv2.BFMatcher()
    try:
        matc=bf.knnMatch(d1,d2,k=2)
        
        for m,n in matc:
                if m.distance < 0.75*n.distance:
                    good.append([m])            
        img=cv2.drawMatchesKnn(ig1,k1,ig2,k2,good,None,flags=2)
        img=cv2.resize(img,(560,440))#resize
        le.append(len(good))
        f=open("log.txt", "a")
        
    
        if(len(le)==3):#max find in no
            ls=(max(le))
            le=[]
            sh=str(ls)
            print(nam+'   :'+sh)#
            
            
            if(ls>thn and ls<thx):
                flg=1
                f.write(str(datetime.datetime.now())+"  :" + nam +'   :'+sh + "   succes  "+str(yy)+"\n")
                #cv2.imwrite(str(datetime.datetime.now())+".jpg",frame)  
            else:#if(ls<th)
                flg=0
                f.write(str(datetime.datetime.now())+"  :" + nam +'   :'+sh + "   move   \n")
        
        if(flg==1):
            cv2.putText(img,na.upper()+'   '+ sh,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)#text
            cv2.putText(img,'.',(490,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),170)#alarm
            #cc=str(yy)+".jpg"
            
            #cv2.imwrite(str(yy)+".jpg",frame) #str(datetime.datetime.now())+".jpg"   "C:/Users/YKDHIv/Desktop/New pro/"+
            #print(cc)
            #cv2.waitKey(0)
                    
        else:
            cv2.putText(img,"move   "+ sh,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)#text
            cv2.putText(img,'.',(490,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),170)#alarm
            cc="out.jpg"
            
            cv2.imwrite(cc,frame)
        f.close()
        
        return (le,flg,img,flg)    
    except:
        matc(ig1,k1,d1,ig2,k2,d2,le,flg,thn,thx,nam,yy)
        print("P")
        
        return (le,flg,img,flg)
        pass        
    

cap=cv2.VideoCapture(1)
kp1,des1=info(imge1)
#cam=cv2.VideoCapture(0)
#kp3,des3=info(imge5)
##converter.say("side ok ok")
while True:
    
    _,frame=cap.read()
    frame = frame[50:250,200:500]
    #_,frame1=cam.read()
    #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    #if frame is not None:
        #frame = frame[250:500,200:450]
    #if frame1 is not None:
     #   frame1 = frame1[100:300,400:700]
    #frame = cv2.erode(frame, kernel, iterations=2) 
    #frame = cv2.filter2D(frame, -2, kernel)
    imge2=frame.copy()
    #frame1 = cv2.erode(frame1, kernel, iterations=2) 
    #frame1 = cv2.filter2D(frame1, -2, kernel)
    #imge4=frame1.copy()
    imge2=cv2.cvtColor(imge2,cv2.COLOR_BGR2GRAY)
    imge2=cv2.Canny(imge2, 100, 200)
    #imge2=cv2.dilate(imge2, kernel, iterations = 1)
    #imge4=cv2.cvtColor(imge4,cv2.COLOR_BGR2GRAY)
    kp2,des2=info(imge2)
    #kp4,des4=info(imge4)
    try:
        lee,fl,oim1,val1=matc(imge2,kp2,des2,imge1,kp1,des1,lee,fl,th1,th2,na,yu)        
        #le2,f2,oim2,val2=matc(imge5,kp3,des3,imge4,kp4,des4,le2,f2,th1,th2,na,yu)
        yu=yu+1
        cv2.imshow('img7',oim1)
        if (cv2.waitKey(1)&0xFF==ord('q')):
            break
    except:
        
        #if(val1==1):# or val2==1):
            #converter.say("side ok ok")
         #   winsound.Beep(2500,100)
            #both=np.concatenate((oim1,oim2),axis=1)
            #both=cv2.resize(both,(1080,640))
        #cv2.imshow('img7',oim1)#both    
        
        pass
f.close() 
cap.release()
cv2.destroyAllWindows()
