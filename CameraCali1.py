import cv2 as cv
import numpy as np
import yaml


# Save the data to yaml
def save(filename, mtx, dist):
    data = {
        "camera_matrix": np.asarray(mtx).tolist(),
        "dist_coeff": np.asarray(dist).tolist(),
    }

    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

# Retrive the data from yaml
def retrive(filename):
    # Retrive the data
    with open("calibration.yaml", "r") as f:
        loadeddict = yaml.load(f)

    mtx_loaded = loadeddict.get("camera_matrix")
    dist_loaded = loadeddict.get("dist_coeff")
    return (mtx_loaded, dist_loaded)

# Calculate reprojection Error
def reprojection_err(objpoints):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))


if __name__ == "__main__":
    # termination criteria to determin the subpixels
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # images = glob.glob('*.jpg')
    cap = cv.VideoCapture(input("path to video with chessboard"))
    count = 10
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            retfind, corners = cv.findChessboardCorners(gray, (7, 6), None)
            # If found, add object points, image points (after refining them)
            if retfind == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                count = count - 1
                if count <= 0:
                    cap.release()
                    break
        else:
            cap.release()
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

    cv.destroyAllWindows()
    # Calculate the interistic data
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"ret:{ret} | mtx:{mtx} | dist:{dist} | rvecs:{rvecs} | tvecs:{tvecs}\n")
    save("data.yaml", mtx, dist)
