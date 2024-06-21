import cv2 as cv
import numpy as np
import yaml
import matplotlib.pyplot as plt

Debug = True


class space:
    def __init__(s, mtx, pt1, pt2, pt3):
        """

        The plane which relevent to the camera in mm units

        
        """
        s.fx = mtx[0, 0]
        s.fy = mtx[1, 1]
        s.cx = mtx[0, 2]
        s.cy = mtx[1, 2]

        # calculate the plane Ax+By+Cz+D=0 N = vec1xvec2
        vec1 = (pt1[0] - pt2[0], pt1[1] - pt2[1], pt1[2] - pt2[2])
        vec2 = (pt1[0] - pt3[0], pt1[1] - pt3[1], pt1[2] - pt3[2])
        s.N = (
            vec1[2] * vec2[3] - vec1[3] * vec2[2],
            vec1[3] * vec2[1] - vec1[1] * vec2[3],
            vec1[1] * vec2[2] - vec1[2] * vec2[1],
        )
        if s.N[2] < 0:
            s.N = (-s.N[0], -s.N[1], -s.N[2])
        s.D = -(s.N[0] * pt1[0] + s.N[1] * pt1[1] + s.N[2] * pt1[2])

    def intersect(s, pixelpt):
        #All units here are in pixels since we do not know either the sensor size or the 
        x = (pixelpt-s.cx)/s.fx
        y = (pixelpt-s.cy)/s.fy
        z = 1

        k = -s.D/(s.N[1]*x+s.N[2]*y+s.N[3]*z)
        return (k*x,k*y,k*z)

        

    """
    Ray is used to describe the light shooting out of the camera, is a pixel to ray representation.
    giving a pixel and a mtx this function returns a ray object which units in mm and coorindates using the camera coordinate
    For the mtx of a given camera is fixed mtx is only needed on init.
    Since the distance on the imagining plane is fX/Z fY/Z the MTX's fx fy describles how many pixels per mm/other units 
    The real distance is pixel/ax 
    """


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
    def __init__(s,rot,tran):
        """
        The rot is the rotation vector and trans is the tranformation vector provided by solving PnP via right thumb law with horizontal as X vertial as Y . The rotation in rad X Y Z axis and only in +pi/-pi using right thumb law to id the rotation and is the rotation of the camera instead of the object as we assume the camera is placed horizontaly thus we only have X rotation, all other rotation is simply put at zero.
        """
        
        

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


def retrieve(filename):
    # Retrieve the data
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


def draw(vecs_list, figure_name):
    m = len(vecs_list[0])
    x = []
    y = []
    z = []

    for i in vecs_list:
        x.append(float(i[0].flatten()))
        y.append(float(i[1].flatten()))
        if m > 2:
            z.append(float(i[2].flatten()))
        else:
            z.append(0.0)
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(x, y, z, label=figure_name)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plt.isinteractive = True
    # constants and global varibles
    fps = 30
    transdata = []
    rotdata = []
    frame_data = []
    reprojected = []

    # Retrieve the camera data
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(
        -1, 3
    )  # The drawn box/axis

    mtx, dist = retrieve(input("The camera profile to use:"))
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
                reprojected.append((tvecs[0], tvecs[2] * np.cos(rvecs[0])))
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
            print("Read frame failed\n")

    # Post processing

    draw(transdata, "The 3d Path")

    kalman = KalmanFilter(1 / fps, 1, 10, 10, 1, 10)

    draw(reprojected, "The projected 2d path")
    pred = []
    update = []
    for i in reprojected:
        pred.append(kalman.predict())
        update.append(kalman.update(i))

    draw(pred, "The predicted path")
    draw(update, "The filtered path")

    with open(input("filename") + ".yaml", "w") as f:
        data = {
            "MTX": mtx.tolist(),
            "Distort": np.asarray(dist).tolist(),
            "Frame": frame_data,
            "Tvecs": np.asarray(transdata).tolist(),
            "Rvecs": np.asarray(rotdata).tolist(),
            "Orginal": np.asarray(reprojected).tolist(),
            "Projected": np.asarray(dist).tolist(),
            "Predicted": np.asarray(pred).tolist(),
            "Filtered": np.asarray(update).tolist(),
        }
        yaml.dump(data, f)
    cv.waitKey()
