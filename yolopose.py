import csv
import cv2
import numpy as np
import yaml
humanpoints = {"5,6":360,"3,4":156,"height":1566,"O1":[(0,0),(360,0),(102,220),(158,220)]}   #6,5,3,4

file = input("the video file to read, csv file should have the name of video file\n")
file = file.lstrip('"')
file = file.rstrip('"')
file_csv = file.split(".")[0]
file_csv = file_csv + ".csv"

csvfile = open(file_csv, newline="\n")
reader = csv.reader(csvfile, delimiter=" ", quotechar="|")

cap = cv2.VideoCapture(file)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))

detectnframe = fps

from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

cv2.namedWindow("Teacher's Area", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Results", cv2.WINDOW_AUTOSIZE)

outputfile = input("the file name to save the pose info:")
outputfile = open(outputfile + ".yaml", "w")

trackingid = 0
counter = 0

for counter, row in enumerate(reader):
    data = row[0].split(",")
    pos1 = (int(data[1]), int(data[2]))  # The upper right corner of the bounding box
    pos2 = (int(data[3]), int(data[4]))  # The size of bounding box
    pos3 = (
        int((2 * pos1[0] + pos2[0]) / 2),
        pos1[1],
    )  # upper middle point of the bouding box
    pos4 = (
        int(data[1]) + int(data[3]),
        int(data[2]) + int(data[4]),
    )  # lower left corner of the bounding box
    if not cap.isOpened():
        print("Error opening video file")
        break
        # Capture frame-by-frame
    ret, frame = cap.read()

    frame_out = cv2.rectangle(
        frame.copy(), pos1, (pos1[0] + pos2[0], pos1[1] + pos2[1]), (0, 200, 000), 3
    )

    frame_out = cv2.rectangle(
        frame_out, pos3, (pos3[0] + 1, pos3[1] + 1), (0, 0, 200), 3
    )
    ratio = 0.4
    cv2.imshow("Teacher's Area", frame_out)

    #This is a work around for no need of looking for who is the teacher In future this effort will be removed 
    Xmin = pos1[0] - int(ratio * pos2[0]) # add ratio*size of box to input of the model
    if Xmin < 0:
        Xmin = 0
    Xmax = pos4[0] + int(ratio * pos2[0])
    if Xmax > frame_width:
        Xmin = frame_width
    Ymin = pos1[1] 
    if Ymin < 0:
        Ymin = 0
    Ymax = pos4[1]                          # Y axis is more stable then X
    if Ymax > frame_height:
        Ymax = frame_height

    target_area = frame[Ymin:Ymax, Xmin:Xmax]

    padded = np.zeros((480, 640, 3), np.uint8)              # the model runs better on image size 480p
    padded[0 : Ymax - Ymin, 0 : Xmax - Xmin] = target_area

    """
    But wait . . . isn't that backward? Shouldn't it instead be image[X,Y]?

    Not so fast!

    Let's back up a step and consider that an image is simply a matrix with a width (number of columns) and height (number of rows). If we were to access an individual location in that matrix, we would denote it as the x value (column number) and y value (row number).

    Therefore, to access the pixel located at x = 50, y = 20, you pass the y-value first (the row number) followed by the x-value (the column number), resulting in image[y, x].

    Note: I've found that the concept of accessing individual pixels with the syntax of image[y, x] is what trips up many. Take a second to convince yourself that image[y, x] is the correct syntax based on the fact that the x-value is your column number (i.e., width), and the y-value is your row number (i.e., height). And we usually access a matrix by [row,col] order.

    Also accessing a reign on matrix is by A[y:y,x:x] = B

    """

    print(f"X{pos1[0]} Y{pos1[1]}\r", end="")
    cv2.imshow("Target_area", padded)

    if counter % detectnframe == 0:

        results = model.track(padded, persist=True)

        track_ids = results[0].boxes.id.int().cpu().tolist()
        npkeypoint = results[0].keypoints.numpy()
        pointlist =[]
        for i, j in enumerate(npkeypoint.data[trackingid]):
            point = (int(j[0])+pos1[1], int(j[1])+pos1[2])
            
            marked = cv2.rectangle(
                frame,
                point
                (point[0]+1,point[1]+1)
                (0, i * 10, 250 - i * 10),
            )
            marked = cv2.putText(
                frame,
                f"{i}",
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, i * 10, 250 - i * 10),
            )
            pointlist.append(point)

        cv2.imshow("Results", marked)
        data = {counter: pointlist}

        yaml.dump(data, outputfile)

        # print(npkeypoint.data[trackingid])
        print(track_ids)

    inputkey = cv2.waitKey(int(1000 / fps)) & 0xFF
    if inputkey == ord("q"):
        print("Aborted!\n")
        cv2.destroyAllWindows()
        outputfile.close()
        break
    elif inputkey == ord(" "):
        print(f"Saving Frame:{counter}")
        if not cv2.imwrite(f"Results{counter}.jpg", marked):
            print("saving img failed\n")


outputfile.close()
