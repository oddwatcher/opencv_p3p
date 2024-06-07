import cv2 as cv
import yaml
import numpy as np
def retrive(filename):
    # Retrive the data
    with open(filename, "r") as f:
        loadeddict = yaml.safe_load(f)

    mtx_loaded = np.asarray(loadeddict.get("camera_matrix"))
    dist_loaded = np.asarray(loadeddict.get("dist_coeff"))

    return (mtx_loaded, dist_loaded)

def undistort(mtx,dist,img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow("undistorted",dst)

mtx = np.zeros((3,3),np.float32)
dist = np.zeros((5,1),np.float32)
if __name__=="__main__" :
    fps = 30
    
    mtx,dist = retrive(input("the profile to use:"))

    print(mtx)
    print(dist)
    src = input("the source of video: ")
    try:
        src = int(src)
        cap = cv.VideoCapture(src)
    except:
        cap = cv.VideoCapture(src)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            cv.imshow("original", img)
            undistort(mtx,dist,img)
            if (cv.waitKey(int(1000/fps))&0xff == ord('q')):
                cv.destroyAllWindows()
                break
            