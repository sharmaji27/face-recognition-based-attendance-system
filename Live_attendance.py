import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import os
import joblib
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
import datetime
import csv
import time
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder


embedding_model = load_model('models/facenet_keras.h5')
print('Embedding Model Loaded')

ML_model = joblib.load('models/face_prediction_model.sav')
print('Loaded ML Model')

detector = MTCNN()

def find_face(img,img_size=(160,160)):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.asarray(img) # converting our image obj to numpy array
    faces = detector.detect_faces(img)
    if faces:
        x,y,w,h = faces[0]['box']
        x,y=abs(x),abs(y)
        face = img[y:y+h,x:x+w]
        face = Image.fromarray(face) # converting it to image object to resize it
        face = face.resize(img_size) # resizing it
        face = np.asarray(face)      # converting it back to array
        return face,x,y,w,h
    return None,None,None,None,None



def embed(face):
    face = face.astype('float32')
    fm,fs = face.mean(),face.std()
    face = (face-fm)/fs
    face = np.expand_dims(face,axis=0)
    embs = embedding_model.predict(face)
    return embs[0]


#for converting ids to names
def id2name(id):
    x = os.listdir('faces/train/')
    return x[id]


def mark_attendance(name,roll):
    
    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')
        
    date=time.asctime()[8:10]
    month=time.asctime()[4:7]
    year=time.asctime()[-4:]
    tim=time.asctime()[11:16]
    
    # if csv of current date doesn't exist, make it
    if (date+'-'+month+'-'+year+'.csv')  not in os.listdir('Attendance/'):
        att = pd.DataFrame(columns=['Roll','Name','Time'])
        att.to_csv('Attendance/'+date+'-'+month+'-'+year+'.csv')
        
    # here we are just selecting these 3 columns everytime and ignoring the index column    
    att = pd.DataFrame(pd.read_csv('Attendance/'+date+'-'+month+'-'+year+'.csv'))
    att = att[['Roll','Name','Time']]
    
    name_list = set(att['Name'])

    if name not in name_list:
        att1 = pd.DataFrame({'Name':[name], 'Roll':[roll], 'Time':[datetime.datetime.now().strftime("%H:%M:%S")]})
        att = att.append(att1,ignore_index=False)
    
    # if the user was already in the attendance list but if it comes again after 5 minute we will add it
    else:
        prev_time = att[att['Name']==name]['Time'].iloc[-1]
        curr_time = datetime.datetime.now().time().strftime("%H:%M:%S")
        #here we are just checking the time difference between previous timestamp and current time
        print(datetime.datetime.strptime(curr_time, '%H:%M:%S') - datetime.datetime.strptime(prev_time, '%H:%M:%S'))
        if datetime.datetime.strptime(curr_time, '%H:%M:%S') - datetime.datetime.strptime(prev_time, '%H:%M:%S') > datetime.timedelta(minutes=5):
            att1 = pd.DataFrame({'Name':[name], 'Roll':[roll], 'Time':[datetime.datetime.now().strftime("%H:%M:%S")]})
            att = att.append(att1,ignore_index=False)

        
    att.to_csv('Attendance/'+date+'-'+month+'-'+year+'.csv')


cap = cv2.VideoCapture(0)
i = 0

while 1:
    i+=1
    ret,frame = cap.read()
    if i>10:
        face,x,y,w,h = find_face(frame)
        if face is not None:
            face_emb = embed(face)
            pred = ML_model.predict(face_emb.reshape(1,-1))
            name = str(id2name(pred[0]))
            if name:
                mark_attendance(name.split('-')[0],name.split('-')[1])
                cv2.rectangle(frame,(x,y),(x+w,y+h),(178,88,239),1)
                cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_TRIPLEX,0.8,(178,88,239),1,cv2.LINE_AA)
        cv2.imshow('live',frame)
    
    if cv2.waitKey(1)==27:
        break
        
cap.release()
cv2.destroyAllWindows()