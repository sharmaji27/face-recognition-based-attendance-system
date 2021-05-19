import cv2
import os
import joblib
import numpy as np
import time
from PIL import Image
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder

######### if these directories doesn't exist, make them #######
try:os.makedirs('faces')
except:pass

try:os.makedirs('faces/train')
except:pass

try:os.makedirs('faces/val')
except:pass


############### take name and roll no as input #################
name = input('Enter your name --> ')
roll = input('Enter your roll no --> ')

name = name+'-'+roll


######### make face folders in train and val ##################
if name in os.listdir('faces/train'):
    print('User already exist in faces')

else:    
    os.makedirs('faces/train/'+name)
    os.makedirs('faces/val/'+name)

    cap = cv2.VideoCapture(0)
    i = 0
    print()
    for i in range(5):
        print(f'Capturing starts in {5-i} seconds...')
        time.sleep(1)
    print('Taking photos...')
    while i<=200:
        ret,frame = cap.read()
        cv2.imshow('taking your pictures',frame)
        if i%5==0 and i<=150 and i!=0:
            cv2.imwrite('faces/train/'+name+'/'+str(i)+'.png',frame)
        elif i%5==0 and i>150:
            cv2.imwrite('faces/val/'+name+'/'+str(i)+'.png',frame)
        i+=1
            
        if cv2.waitKey(1)==27:  #Escape Key
            break

    cv2.destroyAllWindows()
    cap.release()
    print('Successfully taken your photos...')






#### Below part is just training the model with the newly added face. Here we are creating model.

embedding_model = load_model('models/facenet_keras.h5')
print('Embedding Model Loaded')

# making a a mtcnn instance for detecting faces
detector = MTCNN()

def find_face(img,img_size=(160,160)):
    img = cv2.imread(img)
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
        return face
    return None


def embed(face):
    face = face.astype('float32')
    fm,fs = face.mean(),face.std()
    face = (face-fm)/fs # standardizing the data 
    face = np.expand_dims(face,axis=0) # flattening it
    embs = embedding_model.predict(face) # embedding model converts our160*160*3 vector to 128 features
    return embs[0]



def load_dataset(path):
    X = []
    y = []
    for people in os.listdir(path):
        for people_images in os.listdir(path+people):
            face = find_face(path+people+'/'+people_images)
            if face is None:continue
            emb = embed(face)
            X.append(emb)
            y.append(people)
        print('Loaded {} images of {}'.format(len(os.listdir(path+'/'+people)),people)) 
    return np.asarray(X),np.asarray(y)



########### Loading training and testing data using functions defined above #############
print('Loading train data...')
X_train, y_train = load_dataset('faces/train/')

print()

print('Loading test data...')
X_test, y_test = load_dataset('faces/val/')


# l2 normalizing the data
l2_normalizer = Normalizer('l2')

X_train = l2_normalizer.transform(X_train)
X_test  = l2_normalizer.transform(X_test)

#label encoding the y data
label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train)
y_test = label_enc.transform(y_test)


############ Training SVC (Support Vector Classifier) for predicting faces ########
svc = SVC(kernel='linear',probability=True)
svc.fit(X_train,y_train)
joblib.dump(svc,'models/face_prediction_model.sav')
print()

print('SVM Model saved successfully!!')