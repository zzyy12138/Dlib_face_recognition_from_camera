import cv2
import dlib

print(dlib.DLIB_USE_CUDA)
print(dlib.cuda.get_num_devices())
img=cv2.imread("Snipaste_2025-06-18_10-50-45.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_detector=dlib.get_frontal_face_detector()
faces=face_detector(gray)

predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
for face in faces:
    shape=predictor(gray,face)
    for n in range(0,68):
        x=shape.part(n).x
        y=shape.part(n).y
        cv2.circle(img,(x,y),2,(0,0,255),-1)

    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

