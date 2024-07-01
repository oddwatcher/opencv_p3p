import numpy as np

class space:
    def __init__(s, mtx, pt1, pt2, pt3):
        """

        The plane which relevent to the camera in mm units this function returns the place by 3 different points

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
        t = s.D/(s.N[0]^2+s.N[1]^2+s.N[2]^2)
        s.pos = (s.N[0]*t,s.N[1]*t),s.N[2]*t
        print(f"Vec N : x:{s.N[0]} y:{s.N[1]} z:{s.N[2]}", end="\n")
        # to verify the result of the plane u can checkout if Y is near zero since we assume the camera is placed horizontally.

    def __init__(s, rot, tran, mtx):
        """
        The Goal of this init overload is to use a singal frame to determin the classroom floor.
        The rot is the rotation vector and trans is the tranformation vector provided by solving PnP via right thumb law with horizontal as X vertial as Y . The rotation in rad X Y Z axis and only in +pi/-pi using right thumb law to id the rotation and is the rotation of the camera instead of the object as we assume the camera is placed horizontaly thus we only have X rotation, all other rotation is simply put at zero. and P3P assumes the camera is fixed, so the translation and rotation is translation first rotation second.
        We only consider this problem by XZ plane of camera frame. Since we have no idea on where the camera actually is and dont know
        Yet inplemented
        We can only consider things happend in the camera frame since we do not know the relation between real world and the camera.

        """
        s.fx = mtx[0, 0]
        s.fy = mtx[1, 1]
        s.cx = mtx[0, 2]
        s.cy = mtx[1, 2]
        rot = abs(rot[0])
        X = abs(tran[0])
        Y = abs(tran[1])
        Z = abs(tran[2])
        l = np.sqrt(tran[1]^2+tran[2]^2)
        DG = l*np.cos(rot[0]+np.arctan(tran[1]/tran[2]))
        s.pos=(0,DG*np.sin(rot[0]),DG*np.cos(rot[0]))

        s.N = s.Normalize(s.pos)
        
        s.D = -(s.N[0] * tran[0] + s.N[1] * tran[1] + s.N[2] * tran[2])
        print(f"Vec N : x:{s.N[0]} y:{s.N[1]} z:{s.N[2]}", end="\n")
        s.selfpos = ()
    def remapping(s, pixelpt):

        # All units here are in pixels since we do not know either the sensor size or the physical foucous of the camera so we can only use pixel here as unit.
        x = (pixelpt - s.cx) / s.fx
        y = (pixelpt - s.cy) / s.fy
        z = 1

        k = -s.D / (s.N[1] * x + s.N[2] * y + s.N[3] * z) #this calculates the ray and the intersection point
        """
        How does remapping takes place? we use 2 unit vector and the 3d coordinate (all in camera frame)
        The dot product of Tvec and those 2 unit vectors will result in two vecs.
        Which together will represent the 2d-frame coordinate. 
        The 2d frame will X axis paraell to camera X axis and Y axis will be perpendicular to it.
        And their direction is mostly same with the camera frame.
        First We will need to know the origin of the 2d-frame in camera frame hence get the vecs    
        """

        return 1
    
    def Normalize(s,vec):
        l =0
        n =0
        nvec =[]
        for i in vec:
            l+=i^2
            n = n+1
        l = np.sqrt(l)
        
        for i in vec:
            nvec.append(i/l)

        return nvec



if __name__ == "__main__":
    print("test the math",end='\n')
    x = input("X in (a,b,c)")
    print (f"x:{x[0]},{x[1]},{x[2]}")
