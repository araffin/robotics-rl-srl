import cv2
import numpy as np
from matplotlib import pyplot as plt

class MarkerRender(object):
    '''
    Add marker image to the original image. By this way, we can simulate the target and the robot.
    '''

    def __init__(self, noise_var):
        super(MarkerRender, self).__init__()
        self.marker_image_with_margin = None # need to read from file
        self.margin = [0.0,0.0,0.0,0.0]
        
        self.origin_image_shape = None
        self.marker_image_transformed = None
        self.marker_weight_transformed = None
        self.bg_weight = None # back ground mask, inverse of marker_mask

        self.noise_var = noise_var
        
        self.initialized = False

        
    def setMarkerImage(self,marker_image_with_margin, margin):
        if margin is np.ndarray:
            margin = list(margin)
            
        assert len(margin) == 4, "margin should be a list of 4 numbers"
        self.marker_image_with_margin = marker_image_with_margin
        self.margin = margin
        
        self.roi_length = int(np.linalg.norm(self.marker_image_with_margin.shape) + 4)
        if self.roi_length % 2 == 1:
            self.roi_length += 1 # ensure its even
        self.roi_half_length = self.roi_length//2
        
        # setup margin's weight
        self.marker_weight = np.ones(self.marker_image_with_margin.shape[0:3], np.float32)
        for i in range(self.margin[0]):
            self.marker_weight[i,:] = i/self.margin[0] * 0.3
        for i in range(self.margin[1]):
            self.marker_weight[:,i] = i/self.margin[1] * 0.3
        for i in range(self.margin[2]):
            self.marker_weight[-i-1,:] = i/self.margin[2] * 0.3
        for i in range(self.margin[3]):
            self.marker_weight[:,-i-1] = i/self.margin[3] * 0.3
        
    def transformMarkerImage(self, marker_pixel_pos, marker_yaw, maker_scale):
        self.marker_yaw = marker_yaw
        self.marker_pixel_pos = np.float32(marker_pixel_pos).reshape((2,))
        self.maker_scale = maker_scale
        self.M_marker_with_margin =  cv2.getRotationMatrix2D((self.marker_image_with_margin.shape[1]/2,\
                                                              self.marker_image_with_margin.shape[0]/2),\
                                                             self.marker_yaw*180/np.pi,self.maker_scale)
        self.marker_pixel_pos_fraction, self.marker_pixel_pos_interger = np.modf(self.marker_pixel_pos)

        self.marker_pixel_pos_interger = self.marker_pixel_pos_interger.astype(np.int)
        self.M_marker_with_margin[0,2] += self.marker_pixel_pos_fraction[1] + self.roi_half_length - \
                                            self.marker_image_with_margin.shape[0]/2
        self.M_marker_with_margin[1,2] += self.marker_pixel_pos_fraction[0] + self.roi_half_length - \
                                            self.marker_image_with_margin.shape[1]/2

        self.marker_image_transformed = cv2.warpAffine(self.marker_image_with_margin,self.M_marker_with_margin,\
                                                       (self.roi_length,self.roi_length)) 
        
        self.marker_weight_transformed = cv2.warpAffine(self.marker_weight,self.M_marker_with_margin,\
                                                      (self.roi_length,self.roi_length))  # white: Marker part

        self.bg_weight = 1.0 - self.marker_weight_transformed # white: origin image part

    def generateNoise(self):
        mean = 0.0
        noise = np.random.standard_normal(self.marker_image_with_margin.shape) * self.noise_var + mean
        noise = np.around(noise, 0)
        
        return cv2.warpAffine(noise,self.M_marker_with_margin,(self.roi_length,self.roi_length))
        
    def checkBoxIndex(self, box_index):
        '''
        box_index : [x_min, x_max, y_min, y_max]
        '''
        relative_index = [0,box_index[1] - box_index[0],0,box_index[3] - box_index[2]]
        if box_index[0] < 0:
            relative_index[0] = relative_index[0] - box_index[0]
            box_index[0] = 0
        if box_index[1] > self.origin_image_shape[0]:
            
            relative_index[1] = relative_index[1] - (box_index[1] - self.origin_image_shape[0])
            box_index[1] = self.origin_image_shape[0]
        if box_index[2] < 0:
            relative_index[2] = relative_index[2] - box_index[2]
            box_index[2] = 0
        if box_index[3] > self.origin_image_shape[1]:
            relative_index[3] = relative_index[3] - (box_index[3] - self.origin_image_shape[1])
            box_index[3] = self.origin_image_shape[1]
        return box_index, relative_index
    def addMarker(self, origin_image, marker_pixel_pos=None, marker_yaw=None, maker_scale=None):
        
    
        if not self.initialized:
            self.origin_image_shape = origin_image.shape
            self.initialized = True
            
        if marker_pixel_pos is not None:
            # set Marker pixel position
            self.transformMarkerImage(marker_pixel_pos, marker_yaw, maker_scale)

        
        noise = self.generateNoise()
        # combine noise, target, back_ground images togethor
        processed_image = origin_image.copy()
        roi_area = [self.marker_pixel_pos_interger[1] - self.roi_half_length,\
                    self.marker_pixel_pos_interger[1] + self.roi_half_length,
                    self.marker_pixel_pos_interger[0] - self.roi_half_length,\
                    self.marker_pixel_pos_interger[0] + self.roi_half_length]
         
        try:
            processed_image[roi_area[0]:roi_area[1],roi_area[2]:roi_area[3],:] = \
                        ((noise + self.marker_image_transformed) * self.marker_weight_transformed)\
                        + origin_image[roi_area[0]:roi_area[1],roi_area[2]:roi_area[3],:] * self.bg_weight
        except ValueError:
            roi_area, relative_index = self.checkBoxIndex(roi_area) 
            processed_image[roi_area[0]:roi_area[1],roi_area[2]:roi_area[3],:] = \
                    ((noise + self.marker_image_transformed) * self.marker_weight_transformed)\
                    [relative_index[0]: relative_index[1], relative_index[2]:relative_index[3], :] + \
                    origin_image[roi_area[0]:roi_area[1],roi_area[2]:roi_area[3],:] * self.bg_weight[relative_index[0]: relative_index[1], relative_index[2]:relative_index[3], :]
        return processed_image.astype(np.uint8)


if __name__ == "__main__":
    # example, add the Marker image to the observation image
    origin_image = cv2.imread("omnirobot_utils/back_ground.jpg",cv2.IMREAD_COLOR)
    origin_image = cv2.resize(origin_image,(480,480))
    marker_image_with_margin = cv2.imread("omnirobot_utils/robot_margin3_pixel_only_tag.png",cv2.IMREAD_COLOR)
    plt.imshow(origin_image)
    plt.show()
    marker_render = MarkerRender(noise_var=1.0)
    marker_render.setMarkerImage(marker_image_with_margin, [3,3,3,3])

    result = marker_render.addMarker(origin_image, marker_pixel_pos=[470,300], marker_yaw=0, maker_scale=1.0) # row, col, angle(rad) 

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imshow("result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
