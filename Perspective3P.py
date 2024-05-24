import cv2 as cv
import numpy as np
import yaml
import matplotlib.pyplot as plt

fps = 30
criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)  # still needed for feature extraction


def retrive(filename):
    # Retrive the data
    with open(filename, "r") as f:
        loadeddict = yaml.safe_load(f)

    mtx_loaded = np.asarray(loadeddict.get("camera_matrix"))
    dist_loaded = np.asarray(loadeddict.get("dist_coeff"))

    return (mtx_loaded, dist_loaded)


def convertpts(pts_float):
    pts_int = []
    for i in pts_float:
        pts_int.append(tuple(map(int, i.ravel())))
    return pts_int


def drawaxis(img, corners, imgpts):  # Draw the axis
    corner = tuple(map(int, corners[0].ravel()))
    img = cv.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0, 0, 255), 5)
    return img


def getfeaturepoints(
    img: cv.typing.MatLike,
) -> tuple[
    bool, cv.typing.MatLike, cv.typing.MatLike
]:  # return: the img points and the object points
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (4, 3), None)
    objp = np.zeros((3 * 4, 3), np.float32)
    objp[:, :2] = np.mgrid[0:4, 0:3].T.reshape(
        -1, 2
    )  # an array of 42x3 filled with coordinates of points
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return [ret, corners2, objp]
    else:
        return [0, 0, 0]


def drawtranslate(tvecs_list):
    x = []
    y = []
    z = []
    for i in tvecs_list:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(x, y, z, label="3d path on camera coordinate")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    traslatedata = []
    # Retrive the data
    mtx, dist = retrive(input("The camera profile to use:"))

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(
        -1, 3
    )  # The drawn box/axis

    src = input("the image src:")
    try:
        src_num = int(src)
        cap = cv.VideoCapture(src_num)
    except:
        cap = cv.VideoCapture(src)
    frame_count = 0

    while cap.isOpened():

        ret_img, img = cap.read()

        if ret_img:
            frame_count = frame_count + 1
            ret, imgp, objp = getfeaturepoints(img)
            if ret == True:
                ret, rvecs, tvecs = cv.solvePnP(
                    objp, imgp, mtx, dist
                )  # Find the rotation and translation vectors.
                print(
                    f"X:{tvecs[0]} Y:{tvecs[1]} Z:{tvecs[2]} rX:{rvecs[0]} rY:{rvecs[1]} rZ:{rvecs[2]}\r",
                    end="",
                )
                traslatedata.append(tvecs)
                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img = drawaxis(img, imgp, imgpts)

                cv.imshow("img", img)
            else:
                cv.imshow("img", img)

            k = cv.waitKey(int(1000 / fps)) & 0xFF
            if k == ord("s"):
                cv.imwrite(f"output{frame_count}.jpg", img)
            if k == ord("q"):
                cap.release()
                cv.destroyAllWindows()
                break
    drawtranslate(traslatedata)
    cv.waitKey()
