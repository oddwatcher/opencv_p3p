import cv2 as cv
import numpy as np
import yaml

fps = 30

def retrive(filename):
    # Retrive the data
    with open(filename, "r") as f:
        loadeddict = yaml.safe_load(f)

    mtx_loaded = np.asarray(loadeddict.get("camera_matrix"))
    dist_loaded = np.asarray(loadeddict.get("dist_coeff"))

    return (mtx_loaded, dist_loaded)


def drawbox(img, corners, imgpts):  # Draw a box
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def drawaxis(img, corners, imgpts):  # Draw the axis
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


if __name__ == "__main__":

    # Retrive the data
    mtx,dist = retrive(input("The camera profile to use:"))

    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )  # still needed for feature extraction

    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) # an array of 42x3 filled with coordinates of points

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)  #The drawn box/axis 
    src = input("the image src:")
    try:
        src_num = int(src)
        cap = cv.VideoCapture(src_num)
    except:
        cap = cv.VideoCapture(src)
    frame_count = 0
    ret_img,img = cap.read()
    if ret_img:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
        if ret == True:
            frame_count = frame_count+1
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = drawbox(img, corners2, imgpts)
            cv.imshow("img", img)
            k = cv.waitKey(int(1000/fps)) & 0xFF
            if k == ord("s"):
                cv.imwrite(f"output{frame_count}.img", img)
        cv.destroyAllWindows()
