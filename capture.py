# v1 (2024): webcam motion detector. Diffs every frame against the first one,
# boxes anything big that changed, and logs when motion starts and ends.
# Run: uv run --no-project --with opencv-python capture.py   (q to quit)
import csv
import sys
from datetime import datetime

import cv2

first = None
status_list = [0, 0]
rows = []  # [start, end] per motion event
start = None

if sys.platform == "win32":
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    video = cv2.VideoCapture(0)
if not video.isOpened():
    sys.exit("No camera found.")

while True:
    check, frame = video.read()
    if not check:
        break
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first is None:
        first = gray
        continue

    delta = cv2.absdiff(first, gray)
    threshdelta = cv2.threshold(delta, 45, 255, cv2.THRESH_BINARY)[1]
    threshdelta = cv2.dilate(threshdelta, None, iterations=2)

    (cnts, _) = cv2.findContours(threshdelta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        start = datetime.now()
    if status_list[-1] == 0 and status_list[-2] == 1:
        rows.append([start, datetime.now()])
        start = None

    cv2.imshow("delta", delta)
    cv2.imshow("sive", gray)
    cv2.imshow("Tdelta", threshdelta)
    cv2.imshow("pohyb", frame)

    key = cv2.waitKey(200)
    if key == ord("q"):
        break

if start is not None:
    rows.append([start, datetime.now()])

with open("times.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start", "end"])
    writer.writerows(rows)

video.release()
cv2.destroyAllWindows()
