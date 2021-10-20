# face-recognition-based-attendance-system

This Face recognition based attendance system uses MTCNN(Multitask Convolution Neural Networks) for face detection instead of HAAR-CAscades for better detections.  

It detects faces, resize them and converts those into embeddings(vectors).  

I have used SVC here for training on the face embeddings.  

As soon as it recognizes some face in the frame it marks its attendance in the csv file in Attendance folder.  

For adding some new face just run Add_new_face.py and enter your name and roll no.  

For detections just run Live_Attendance.py  
 
PS: Please don't compare this face recognition model with the HAAR-Cascade face detector.  

Do visit my blog for better explanations: https://machinelearningprojects.net/face-recognition-based-attendance-system/
