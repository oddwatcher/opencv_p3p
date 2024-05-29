import cv2 as cv
import numpy as np
import yaml
import matplotlib.pyplot as plt

Debug = False

class KalmanFilter(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
         dt: sampling time (time for 1 cycle)
         u_x: acceleration in x-direction
         u_y: acceleration in y-direction
         std_acc: process noise magnitude
         x_std_meas: standard deviation of the measurement in x-direction
         y_std_meas: standard deviation of the measurement in y-direction
        """
        # Define sampling time
        self.dt = dt
        # Define the  control input variables
        self.u = np.matrix([[u_x], [u_y]])
        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])
        # Define the State Transition Matrix A
        self.A = np.matrix(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        # Define the Control Input Matrix B
        self.B = np.matrix(
            [[(self.dt**2) / 2, 0], [0, (self.dt**2) / 2], [self.dt, 0], [0, self.dt]]
        )
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
        # Initial Process Noise Covariance
        self.Q = (
            np.matrix(
                [
                    [(self.dt**4) / 4, 0, (self.dt**3) / 2, 0],
                    [0, (self.dt**4) / 4, 0, (self.dt**3) / 2],
                    [(self.dt**3) / 2, 0, self.dt**2, 0],
                    [0, (self.dt**3) / 2, 0, self.dt**2],
                ]
            )
            * std_acc**2
        )
        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0], [0, y_std_meas**2]])
        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  
        # Update time state
        # x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  # Eq.(12)
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)
        return self.x[0:2]



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
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )  # still needed for feature extraction

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


def draw(vecs_list):
    m = len(vecs_list[0])
    x = []
    y = []
    z = []

    for i in vecs_list:
        x.append(float(i[0]))
        y.append(float(i[1]))
        if m>2:
            z.append(float(i[2]))
        else:
            z.append(0.0)
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(x, y, z, label="3d path on camera coordinate")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plt.isinteractive=True
    # constants
    fps = 30
    transdata = []
    rotdata=[]
    frame_data=[]

    # Retrive the camera data
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(
        -1, 3
    )  # The drawn box/axis
    
    mtx, dist = retrive(input("The camera profile to use:"))
    # End of constants
    
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

                transdata.append(tvecs)
                rotdata.append(rvecs)
                frame_data.append(frame_count)

                # project 3D points to image plane
                if Debug:
                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                    img = drawaxis(img, imgp, imgpts)
                    cv.imshow("img", img)
            else:
                if Debug:
                    cv.imshow("img", img)

            k = cv.waitKey(int(1000 / fps)) & 0xFF
            if k == ord("s"):
                cv.imwrite(f"output{frame_count}.jpg", img)
            if k == ord("q"):
                cap.release()
                cv.destroyAllWindows()
                break
        else:
            print("read frame failed\n")

    # Post processing

    draw(transdata)
    kalman=KalmanFilter(1/fps, 1, 10, 10, 1,1)

    reprojected=[]
    for n,i in enumerate(transdata):
        reprojected.append((i[0],i[2]*np.cos(rotdata[n][0])))
    draw(reprojected)
    pred = []
    update = []
    for i in reprojected:
        pred.append(kalman.predict())
        update.append(kalman.update(i))

    draw(pred)
    draw(update)
    cv.waitKey()
