import cv2,time,pandas
from datetime import datetime
first=None
status_list=[None, None]
times=[]
df=pandas.DataFrame(columns=["start", "end"])
video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
p=1
while p!=0:
    check, frame = video.read()
    status=0





    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21, 21), 0)

    if first is None:
        first=gray
        continue



    delta=cv2.absdiff(first,gray)
    threshdelta=cv2.threshold(delta, 45, 255, cv2.THRESH_BINARY)[1]

    threshdelta=cv2.dilate(threshdelta, None, iterations=2)


    (cnts,_) = cv2.findContours(threshdelta.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0,255,0), 3)



    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
        
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow("delta",delta)
    cv2.imshow('sive', gray  )
    cv2.imshow("Tdelta",threshdelta)
    cv2.imshow("pohyb", frame)



    key=cv2.waitKey(200)
    if key==ord('q'):
        break
print(status_list)
print(times)
df.to_csv("times.csv")
video.release()
cv2.destroyAllWindows()