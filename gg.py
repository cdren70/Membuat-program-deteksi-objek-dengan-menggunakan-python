import cv2 as cv

cam = cv.VideoCapture(0)

eye_detector = cv.CascadeClassifier('D:\Latihan Coding\Tugas_Akhir\src\haarcascade_eye (1).xml')
smile_detector = cv.CascadeClassifier('D:\Latihan Coding\Tugas_Akhir\src\haarcascade_smile.xml')
face_detector = cv.CascadeClassifier('D:\Latihan Coding\Tugas_Akhir\src\haarcascade_frontalface_alt.xml')

while True :
    retV, frame = cam.read()
    warna = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    eye = eye_detector.detectMultiScale(warna, 1.1, 8)
    smile = smile_detector.detectMultiScale(warna, 1.1, 8)
    face = face_detector.detectMultiScale(warna, 1.1, 8)
    
    for x, y, w, h in eye :
        cv.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0),
            1)
    for x, y, w, h in smile :
        cv.rectangle(
            frame, (x, y), (x + w, y + h),
            (255, 0, 0),
            1)
    for x, y, w, h in face :
        cv.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 0, 255),
            5)
        
    res_frame = cv.resize(frame, (1280,960))   
    cv.imshow('Hasilnya',res_frame)    
    if cv.waitKey(1) == ord('x'):
        break

cam.release()
cv.destroyAllWindows()
            
    


