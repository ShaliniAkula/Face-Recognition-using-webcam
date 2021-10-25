import numpy as np
import cv2
import os
from PIL import Image
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QInputDialog
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize    
import sqlite3

def faceAdd(self):
    if not os.path.exists('images'):
        os.makedirs('images')
        
    #Capturing the images
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,480)
    count = 0
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    text, ok=QInputDialog.getText(self,"Input Dialog","Enter User name")
    if ok:
        try:
            #Fetching the id from DB
            Details=sqlite3.connect('data.db')
            CurDetails=Details.cursor()
            CurDetails.execute("SELECT MAX(ID) FROM Details;")
            row = CurDetails.fetchone()
            if not row[0] == None:
                face_id=int(row[0])+1
            else:
                face_id=1

            name=text
            print("\n [INFO] Initializing face capture. Look the camera and wait ...")
            while(True):
                #converting the images into grayscale
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                    count += 1
                    cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('image', img)
                k = cv2.waitKey(100) & 0xff
                if k < 30:
                    break
                elif count >= 100:
                     break

            print("\n [INFO] Exiting Program.")
            cam.release()
            cv2.destroyAllWindows()

            """train data"""

            path = './images/'
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
            
            def getImagesAndLabels(path):
                imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
                faceSamples=[]
                ids = []
                #training the algorithm with haarcascade classifier
                for imagePath in imagePaths:
                    print(os.path)
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img,'uint8')
                    id = int(os.path.split(imagePath)[-1].split(".")[1])
                    faces = detector.detectMultiScale(img_numpy)
                    for (x,y,w,h) in faces:
                        faceSamples.append(img_numpy[y:y+h,x:x+w])
                        ids.append(id)
                return faceSamples,ids
            print ("\n[INFO] Training faces...")
            
            faces,ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))
            recognizer.write('trainer.yml')

            CurDetails.execute("INSERT INTO Details (ID,NAME)VALUES(?,?);",(face_id,name))
            Details.commit()
            
        except sqlite3.Error as error:
            print("Failed to execute the query ", error)
        finally:
            if (Details):
                Details.close()
        print("\n[INFO] {0} faces trained.".format(len(np.unique(ids))))
        faceRecog(self)
    return

def faceRecog(self):
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    face_cascade_Path = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(face_cascade_Path)
    font = cv2.FONT_HERSHEY_SIMPLEX 

    id = 0
    names = ['None']
    try:
        Details=sqlite3.connect('data.db')
        CurDetails=Details.cursor()
        CurDetails.execute("SELECT Name FROM Details ORDER BY ID;")
        records=CurDetails.fetchall()
        for row in records:
            names.append(row[0])
        
    except sqlite3.Error as error:
        print("Failed to execute the query ", error)
    finally:
        if (Details):
            Details.close()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
              
                id = "Who are you ?"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x -30, y + h - 5), font, 1, (154,205,50), 2)

        cv2.imshow('camera', img)
        
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    return

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        self.m_w11 = QtWidgets.QWidget()
        self.m_w12 = QtWidgets.QWidget()
        self.m_w21 = QtWidgets.QWidget()
        self.m_w22 = QtWidgets.QWidget()

        lay = QtWidgets.QGridLayout(central_widget)

        for w, (r, c) in zip(
            (self.m_w11, self.m_w12, self.m_w21, self.m_w22),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        ):
            lay.addWidget(w, r, c)
        for c in range(2):
            lay.setColumnStretch(c, 1)
        for r in range(2):
            lay.setRowStretch(r, 1)

        lay = QtWidgets.QVBoxLayout(self.m_w11)

        self.setMinimumSize(QSize(1366 , 768))    
        self.setWindowTitle("Face Recognizer") 

        pybutton = QPushButton('Add New Face', self)
        pybutton.clicked.connect(self.clickMethodAdd)
        pybutton.resize(200,32)
        pybutton.move(550, 250)

        pybutton1 = QPushButton('Recognize', self)
        pybutton1.clicked.connect(self.clickMethodRecog)
        pybutton1.resize(200,32)
        pybutton1.move(550, 450)
        
        #lay = QtWidgets.QVBoxLayout(self.m_w21)
        #Status = QLabel('Status', self)
        #lay.addWidget(Status)

    def clickMethodAdd(self):
        faceAdd(self)
    def clickMethodRecog(self):
        faceRecog(self)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())




