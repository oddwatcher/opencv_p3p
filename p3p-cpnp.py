import cv2 as cv
import numpy as np
import yaml
import matplotlib.pyplot as plt

Debug = True

from numpy import linalg
from scipy.linalg import expm, eigh, eig, svd


def CPnP(s, Psens_2D, fx, fy, u0, v0):
    """
    Inputs: s - a 3xn matrix whose i-th column is the coordinates (in the world frame) of the i-th 3D point  

    Psens_2D - a 2xn matrix whose i-th column is the coordinates of the 2D projection of the i-th 3D point  

    fx, fy, u0, v0 - intrinsics of the camera, corresponding to the intrinsic matrix K=[fx 0 u0;0 fy v0;0 0 1]  

    Outputs: R - the estimate of the rotation matrix in the first step  

    t - the estimate of the translation vector in the first step  
    
    R_GN - the refined estimate of the rotation matrix with Gauss-Newton iterations  
    
    t_GN - the refined estimate of the translation vector with Gauss-Newton iterations  
    """
    N = s.shape[1]
    bar_s = np.mean(s, axis=1).reshape(3, 1)
    Psens_2D = Psens_2D - np.array([[u0], [v0]])
    obs = Psens_2D.reshape((-1, 1), order="F")
    pesi = np.zeros((2 * N, 11))
    G = np.ones((2 * N, 1))
    W = np.diag([fx, fy])
    M = np.hstack(
        [
            np.kron(bar_s.T, np.ones((2 * N, 1))) - np.kron(s.T, np.ones((2, 1))),
            np.zeros((2 * N, 8)),
        ]
    )

    for k in range(N):
        pesi[[2 * k], :] = np.hstack(
            [
                -(s[0, k] - bar_s[0]) * obs[2 * k],
                -(s[1, k] - bar_s[1]) * obs[2 * k],
                -(s[2, k] - bar_s[2]) * obs[2 * k],
                (fx * s[:, [k]]).T.tolist()[0],
                fx,
                0,
                0,
                0,
                0,
            ]
        )
        pesi[[2 * k + 1], :] = np.hstack(
            [
                -(s[0, k] - bar_s[0]) * obs[2 * k + 1],
                -(s[1, k] - bar_s[1]) * obs[2 * k + 1],
                -(s[2, k] - bar_s[2]) * obs[2 * k + 1],
                0,
                0,
                0,
                0,
                (fy * s[:, [k]]).T.tolist()[0],
                fy,
            ]
        )

    J = np.dot(np.vstack([pesi.T, obs.T]), np.hstack([pesi, obs])) / (2 * N)
    delta = np.vstack(
        [
            np.hstack([np.dot(M.T, M), np.dot(M.T, G)]),
            np.hstack([np.dot(G.T, M), np.dot(G.T, G)]),
        ]
    ) / (2 * N)

    w, D = eig(J, delta)
    sigma_est = min(abs(w))

    est_bias_eli = np.dot(
        np.linalg.inv((np.dot(pesi.T, pesi) - sigma_est * (np.dot(M.T, M))) / (2 * N)),
        (np.dot(pesi.T, obs) - sigma_est * np.dot(M.T, G)) / (2 * N),
    )
    bias_eli_rotation = np.vstack(
        [est_bias_eli[3:6].T, est_bias_eli[7:10].T, est_bias_eli[0:3].T]
    )
    bias_eli_t = np.hstack(
        [
            est_bias_eli[6],
            est_bias_eli[10],
            1
            - bar_s[0] * est_bias_eli[0]
            - bar_s[1] * est_bias_eli[1]
            - bar_s[2] * est_bias_eli[2],
        ]
    ).T
    normalize_factor = np.linalg.det(bias_eli_rotation) ** (1 / 3)
    bias_eli_rotation = bias_eli_rotation / normalize_factor
    t = bias_eli_t / normalize_factor

    U, x, V = svd(bias_eli_rotation)
    V = V.T

    RR = np.dot(U, np.diag([1, 1, np.linalg.det(np.dot(U, V.T))]))
    R = np.dot(RR, V.T)

    E = np.array([[1, 0, 0], [0, 1, 0]])
    WE = np.dot(W, E)
    e3 = np.array([[0], [0], [1]])
    J = np.zeros((2 * N, 6))

    g = np.dot(WE, np.dot(R, s) + np.tile(t, N).reshape(N, 3).T)
    h = np.dot(e3.T, np.dot(R, s) + np.tile(t, N).reshape(N, 3).T)

    f = g / h
    f = f.reshape((-1, 1), order="F")
    I3 = np.diag([1, 1, 1])

    for k in range(N):
        J[[2 * k, 2 * k + 1], :] = (
            np.dot(
                (WE * h[0, k] - g[:, [k]] * e3.T),
                np.hstack(
                    [
                        s[1, k] * R[:, [2]] - s[2, k] * R[:, [1]],
                        s[2, k] * R[:, [0]] - s[0, k] * R[:, [2]],
                        s[0, k] * R[:, [1]] - s[1, k] * R[:, [0]],
                        I3,
                    ]
                ),
            )
            / h[0, k] ** 2
        )

    initial = np.hstack([np.zeros((3)), t.tolist()]).reshape(6, 1)
    results = initial + np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), (obs - f))
    X_GN = results[0:3]
    t_GN = results[3:6]
    Xhat = np.array(
        [[0, -X_GN[2], X_GN[1]], [X_GN[2], 0, -X_GN[0]], [-X_GN[1], X_GN[0], 0]]
    )
    R_GN = np.dot(R, expm(Xhat))

    return R, t, R_GN, t_GN


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


def draw(vecs_list):
    m = len(vecs_list[0])
    x = []
    y = []
    z = []

    for i in vecs_list:
        x.append(float(i[0]))
        y.append(float(i[1]))
        if m > 2:
            z.append(float(i[2]))
        else:
            z.append(0.0)
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(x, y, z, label="3d path on camera coordinate")
    ax.legend()
    plt.show()

def undistort(mtx,dist,img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    return dst


if __name__ == "__main__":
    plt.isinteractive = True
    # constants and global varibles
    fps = 30
    framelist = []
    imgpdata = []
    objpdata = []
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
            img = undistort(mtx,dist,img)
            
            frame_count = frame_count + 1
            ret, imgp, objp = getfeaturepoints(img)

            if ret == True:
                imgpdata.append(imgp)
                objpdata.append(objp)
                framelist.append(frame_count)
                if Debug:
                    cv.drawChessboardCorners(img,(3,3),imgp)
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

    imgpmat = np.asarray(imgpdata,np.float32,'C',(3,-1))
    objpmat = np.asarray(objpdata,np.int32,'C',(3,-1))
    mtxlist = mtx.flatten()

    R, t, R_GN, t_GN = CPnP(objpmat,imgpmat,mtxlist[0],mtxlist[4],mtxlist[2],mtxlist[5])

    kalman = KalmanFilter(1 / fps, 1, 10, 100, 1, 10)

    pred = []
    update = []

    for i in reprojected:
        pred.append(kalman.predict())
        update.append(kalman.update(i))

    draw(pred)
    draw(update)
    cv.waitKey()
