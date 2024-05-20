import cv2 as cv
import numpy as np
import yaml


# Save the data to yaml
def save(filename, mtx, dist):
    data = {
        "camera_matrix": np.asarray(mtx).tolist(),
        "dist_coeff": np.asarray(dist).tolist(),
    }

    with open(filename+".yaml", "w") as f:
        yaml.dump(data, f)

# Retrive the data from yaml
def retrive(filename):
    # Retrive the data
    with open(filename, "r") as f:
        loadeddict = yaml.load(f)

    mtx_loaded = loadeddict.get("camera_matrix")
    dist_loaded = loadeddict.get("dist_coeff")
    return (mtx_loaded, dist_loaded)

# Calculate reprojection Error
def reprojection_err(objpoints, mtx, dist, rvecs, tvecs):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))
    return mean_error / len(objpoints)

# Try if new matrix works
def undistort(mtx,dist,img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    cv.imshow("undistorted",dst)

if __name__ == "__main__":
    # termination criteria to determin the subpixels
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) 
    # the object points already 
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) #(42 points 42:3) and their coordinate is known
    print(f'array shape:{objp.shape} {objp}')

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # images = glob.glob('*.jpg')
    vidpath = input("path to video with chessboard")
    try:
        vidnum = int(vidpath)
        cap = cv.VideoCapture(vidnum)
    except:
        cap = cv.VideoCapture(vidpath)
    count = 10
    framecount =0
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            cv.imshow("img", img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            retfind, corners = cv.findChessboardCorners(gray, (7, 6), None)
            # If found, add object points, image points (after refining them)
            framecount = framecount+1
            print(f"Current frame:{framecount}")
            if retfind == True:
                objpoints.append(objp)      #Make place for new object points
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                print("corners found\n")
                # Draw and display the corners
                cv.drawChessboardCorners(img, (7, 6), corners2, retfind)
                cv.imshow("corners", img)

                # Only need ten valid frames
                count = count - 1
                if count <= 0:
                    cap.release()
                    break
            cv.waitKey(500)
        else:
            cap.release()
    
    cv.destroyAllWindows()
    # Calculate the interistic data
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"ret:{ret} \n mtx:{mtx} \n dist:{dist} \n rvecs:{rvecs} \n tvecs:{tvecs}\n")
    result = input("the name to save")
    save(result, mtx, dist)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))

    undistort(mtx,dist,img)
    imgdis = np.zeros_like(img)
    newmtx = np.zeros_like(mtx)
    cv.imshow("undistort",imgdis)
    cv.waitKey()

