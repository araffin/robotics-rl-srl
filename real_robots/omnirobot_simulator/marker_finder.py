import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
import yaml

def rotateMatrix90(matrix):
    new_matrix = np.transpose(matrix)
    new_matrix = np.flip(new_matrix, axis=1)
    return new_matrix
            
def hammingDistance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

class MakerFinder():
    def __init__(self, camera_info_path):
        self.min_area = 70
        self.marker_code = {}
        with open(camera_info_path, 'r') as stream:
            try:
                contents = yaml.load(stream)
                camera_matrix = np.array(contents['camera_matrix']['data'])
                self.origin_size = np.array([contents['image_width'], contents['image_height']])
                self.camera_matrix = np.reshape(camera_matrix,(3,3))
                self.distortion_coefficients = np.array(contents['distortion_coefficients']['data'])
            except yaml.YAMLError as exc:
                print(exc)
    def setMarkerCode(self, marker_id, marker_code, real_length):
        self.marker_code[marker_id] = (np.zeros((4,*marker_code.shape[0:2])))
        
        self.marker_code[marker_id][0,:,:] = marker_code
        for i in range(1,4):
            self.marker_code[marker_id][i,:,:] = rotateMatrix90(self.marker_code[marker_id][i-1,:,:])
        self.marker_rows, self.marker_cols = 90, 90
        self.marker_square_pts = np.float32([[0,0],[self.marker_rows,0], [self.marker_rows,self.marker_cols],[0,self.marker_cols]\
                                            ]).reshape(-1,1,2)
        self.marker_real_corners =  np.float32([[0,0,0],[real_length,0,0], [real_length,real_length,0],\
                                                [0,real_length,0]]) - np.float32([real_length/2.0, real_length/2.0,0])
        self.marker_real_corners = self.marker_real_corners.reshape(-1,1,3)
    def intersection(self, l1, l2):
        vx = l1[0]
        vy = l1[1]
        ux = l2[0]
        uy = l2[1]
        wx = l2[2]-l1[2]
        wy = l2[3]-l1[3]

        tmp = vx*uy-vy*ux
        if tmp==0:
            tmp = 1

        #if(/*tmp <= 1.f && tmp >= -1.f && */tmp != 0.f && ang > 0.1)
        s = (vy*wx-vx*wy) / (tmp)
        px = l2[2]+s*ux
        py = l2[3]+s*uy


        return (px, py)
    
    def checkBorder(self, contour, width, height):
        ret = True
        for pt in contour:
            pt = pt[0]
            if((pt[0] <= 1) or (pt[0] >= width-2) or (pt[1] <= 1) or (pt[1] >= height-2)):
                ret = False
        return ret
    
    def labelSquares(self, img_input, visualise:bool):
        img = cv2.resize(img_input, (480,480))
        self.cropped_margin = (self.origin_size - np.array(img.shape[0:2]))/2.0
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.edge = cv2.adaptiveThreshold(self.gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV,31,5)

        _,cnts,_ = cv2.findContours(self.edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        candidate_contours = []
        candidate_approx = []

        for contour in cnts:
            if(len(contour) < 50): #filter the short contour
                continue
            approx = cv2.approxPolyDP(contour, cv2.arcLength(contour,True)*0.035, True)
            if len(approx)==4 and self.checkBorder(approx,self.gray.shape[1], self.gray.shape[0]) \
                and abs(cv2.contourArea(approx)) > self.min_area and cv2.isContourConvex(approx):
                    cv2.drawContours(img, approx, -1, (0,255,0), 3)
                    candidate_approx.append(approx)
                    candidate_contours.append(contour)
        
        n_blob = len(candidate_approx)
        self.blob_corners = np.zeros((n_blob, 4,2), np.float32)

        for i in range(n_blob):
            #find how much points are in each line (length)
            fitted_lines = np.zeros((4,4), np.float)
            for j in range(4):
                pt0 = candidate_approx[i][j]
                pt1 = candidate_approx[i][(j+1)%4]
                k0=-1
                k1=-1
                # find corresponding approximated point (pt0, pt1) in contours
                for k in range(len(candidate_contours[i])):
                    pt2 = candidate_contours[i][k]
                    if pt2[0][0] == pt0[0][0] and pt2[0][1] == pt0[0][1]:
                        k0=k
                    if pt2[0][0] == pt1[0][0] and pt2[0][1] == pt1[0][1]:
                        k1=k

                # compute how much points are in this line
                if(k1>=k0):
                    length = k1-k0-1
                else:
                    length = len(candidate_contours[i]) - k0+k1-1
                if length == 0:
                    length = 1

                line_pts = np.zeros((1,length,2),np.float32)
                # append this line's point to array 'line_pts'
                for l in range(length):
                    ll = (k0+l+1)%len(candidate_contours[i])
                    line_pts[0,l,:] = candidate_contours[i][ll]

                # Fit edge and put to vector of edges
                [vx,vy,x,y] = cv2.fitLine(line_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                fitted_lines[j,:] = [vx,vy,x,y]
                if visualise:
                    #Finally draw the line
                    # Now find two extreme points on the line to draw line
                    left = [0,0]
                    right =[0,0]
                    length = 100
                    left[0] = x - vx * length
                    left[1] = y - vy * length
                    right[0] = x + vx * length
                    right[1] = y + vy * length
                    cv2.line(img,tuple(left),tuple(right),255,2)

            # Calculated four intersection points
            for j in range(4):
                intc = self.intersection(fitted_lines[j,:],fitted_lines[(j+1)%4,:]);
                self.blob_corners[i,j,:] = intc;
            if visualise:
                for j in range(4):
                    intc = tuple(self.blob_corners[i,j,:]);
                    print(intc)
                    if j == 0:
                        cv2.circle(img, intc, 5, (255, 255, 255))
                    if (j == 1):
                        cv2.circle(img, intc, 5, (255, 0, 0))
                    if (j == 2):
                        cv2.circle(img, intc, 5, (0, 255, 0))
                    if (j == 3):
                        cv2.circle(img, intc, 5, (0, 0, 255))
                cv2.imshow('frame', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    def decode(self, transformed_img):
        num_step =9
        code = np.zeros((num_step,num_step), np.uint8)
        step = int(transformed_img.shape[0]/num_step)
        print(int(step))
        for i in range(num_step):
            for j in range(num_step):
                if np.average(transformed_img[i*step:i*step+step,j*step:j*step+step]) < 100:
                    #print("i,j:",i," ",j," average: ",np.average(transformed_img[i*step:i*step+step,j*step:j*step+step]))
                    
                    #plt.imshow(cv2.cvtColor(transformed_img[i*step:i*step+step,j*step:j*step+step], cv2.COLOR_GRAY2RGB))
                    #plt.show()
                    code[i,j] = 1
                else:
                    code[i,j] = 0
        return code
                
    def rotateCorners(self, corner, i):
        return np.array([corner[i%4,:], corner[(i+1)%4,:], corner[(i+2)%4,:],corner[(i+3)%4,:]])
    def setMarkerImg(self, img):
        self.marker_img = img
    def findMarker(self, marker_id, visualise=False):
        
        
        for i_square in range(self.blob_corners.shape[0]):
            # create marker's mask
            self.marker_mask = np.zeros(self.gray.shape[0:3],np.uint8)
            cv2.fillConvexPoly( self.marker_mask, self.blob_corners[i_square,:,:].astype(np.int32).reshape(-1,1,2), 255);
            
            H , _ = cv2.findHomography(self.blob_corners[i_square,:,:], self.marker_square_pts, cv2.LMEDS)

            marker_transformed = cv2.warpPerspective(self.edge, H, self.gray.shape[0:2])[0:self.marker_rows, 0:self.marker_cols]
            if visualise:
                cv2.imshow('frame', marker_transformed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            code = self.decode(marker_transformed)
            distance = np.zeros(4)
            for i_rotate in range(4): # determinate 0, 90, 180, 270 degree
                distance[i_rotate] = hammingDistance(code.reshape((-1,)), self.marker_code[marker_id][i_rotate,:,:].reshape((-1,)))
            dist_min_arg = np.argmin(distance)
            if visualise:
                print("minimum hangming distance: ",distance[dist_min_arg])
                print(code)
            if distance[dist_min_arg] < 3:
                # find the correct marker!
                # rotate the corners to the correct angle
                self.blob_corners[i_square,:,:] = self.rotateCorners(self.blob_corners[i_square,:,:], dist_min_arg)
                # compute the correct H
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                self.blob_corners[i_square,:,:] = cv2.cornerSubPix(self.gray,self.blob_corners[i_square,:,:],(11,11),(-1,-1),criteria)
                # Find the rotation and translation vectors. (camera to tag)
                print(self.marker_real_corners.shape)
                print(self.blob_corners[i_square,:,:].shape)
                print("margin",self.cropped_margin)
                _, rot_vec, trans_vec = cv2.solvePnP(self.marker_real_corners, self.blob_corners[i_square,:,:]+ self.cropped_margin, \
                                                         self.camera_matrix.astype(np.float32), self.distortion_coefficients.astype(np.float32))
                rot_mat,_ = cv2.Rodrigues(rot_vec)
                # reverse the rotation and translation (tag to camera)
                tag_trans_coord_cam = - trans_vec
                tag_rot_coord_cam = np.linalg.inv(rot_mat)
                
                
                if visualise:
                    marker_img_corner = np.float32([[0,0], [self.marker_img.shape[0],0], \
                                                    [self.marker_img.shape[0],self.marker_img.shape[1]],\
                                                    [0,self.marker_img.shape[1]]]).reshape(-1,1,2)
                    
                    H , _ = cv2.findHomography(marker_img_corner,self.blob_corners[i_square,:,:] + self.cropped_margin)
                    print(self.blob_corners[i_square,:,:] + self.cropped_margin)
                    print(self.blob_corners[i_square,:,:])
                    reconst = cv2.warpPerspective(self.marker_img,H,self.gray.shape)
                    cv2.imshow('frame', reconst)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                return tag_trans_coord_cam, tag_rot_coord_cam

    def getMarkerPose(self, img, marker_ids, visualise=False):
        self.labelSquares(img,visualise)
        if marker_ids is list:
            pos_array = []
            rot_array = []
            for i in marker_ids:
                pos, rot = self.findMarker(marker_id)
                pos_array.append(pos)
                rot_array.append(rot)
            return pos_array, rot_array
        else:
            return self.findMarker(marker_id)

def transformPosCamToGround(pos_coord_cam):
    pos_coord_ground = pos_coord_cam
    pos_coord_ground[0] =  pos_coord_ground[1]
    pos_coord_ground[1] = - pos_coord_ground[0]
    pos_coord_ground[2] = 0
    return pos_coord_cam


if __name__ == "__main__":
    # example, find the markers in the observation image, and get it's position in the ground
    camera_info_path = "/home/gaspard/Documents/ros_omnirobot/catkin_ws/src/omnirobot-dream/omnirobot_remote/cam_calib_info.yaml"
    path = "/home/gaspard/Documents/ros_omnirobot/robotics-rl-srl-omnirobot/data/omnirobot_real_20190125_155902/"
    img = cv2.imread(join(path,"record_008/frame000015.jpg"))
    robot_tag_img = cv2.imread("robot_tag.png")
    robot_tag_code = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    maker_finder = MakerFinder(camera_info_path)
    maker_finder.setMarkerCode('robot',robot_tag_code, 0.18) # marker size 0.18m * 0.18m
    maker_finder.setMarkerImg(robot_tag_img)
    pos_coord_cam, rot_coord_cam = maker_finder.getMarkerPose(img, 'robot',False) 
    
    pos_coord_ground = transformPosCamToGround(pos_coord_cam)
    print("position in the ground: ", pos_coord_cam)