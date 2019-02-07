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
        self.marker_weight_trasnformed = None
        self.bg_weight = None # back ground mask, inverse of marker_mask

        self.noise_var = noise_var
        
        self.initialized = False
        
    def setMarkerImage(self,marker_image_with_margin, margin):
        if margin is np.ndarray:
            margin = list(margin)
            
        assert len(margin) == 4, "margin should be a list of 4 numbers"
        self.marker_image_with_margin = marker_image_with_margin
        self.margin = margin
        
        
    def transformMarkerImage(self, marker_pixel_pos, marker_yaw, maker_scale):
        self.marker_yaw = marker_yaw
        self.marker_pixel_pos = marker_pixel_pos
        self.maker_scale = maker_scale
        self.M_marker_with_margin =  cv2.getRotationMatrix2D((self.marker_image_with_margin.shape[1]/2,\
                                                              self.marker_image_with_margin.shape[0]/2),\
                                                             self.marker_yaw*180/np.pi,self.maker_scale)
        
        
        self.M_marker_with_margin[0,2] += self.marker_pixel_pos[0] - self.marker_image_with_margin.shape[0]/2
        self.M_marker_with_margin[1,2] += self.marker_pixel_pos[1] -  self.marker_image_with_margin.shape[1]/2

        # setup margin's weight
        marker_weight = np.ones(self.marker_image_with_margin.shape[0:3], np.float32)
        for i in range(self.margin[0]):
            marker_weight[i,:] = i/self.margin[0] * 0.3
        for i in range(self.margin[1]):
            marker_weight[:,i] = i/self.margin[1] * 0.3
        for i in range(self.margin[2]):
            marker_weight[-i-1,:] = i/self.margin[2] * 0.3
        for i in range(self.margin[3]):
            marker_weight[:,-i-1] = i/self.margin[3] * 0.3
            
        self.marker_image_transformed = cv2.warpAffine(self.marker_image_with_margin,self.M_marker_with_margin,\
                                                       (self.origin_image_shape[1],self.origin_image_shape[0])) 
        
        self.marker_weight_trasnformed = cv2.warpAffine(marker_weight,self.M_marker_with_margin,\
                                                      (self.origin_image_shape[1],self.origin_image_shape[0]))  # white: Marker part
        
        
        self.bg_weight = 1.0 - self.marker_weight_trasnformed # white: origin image part

    def generateNoise(self):
        mean = 0.0
        noise = np.random.standard_normal(self.marker_image_with_margin.shape) * self.noise_var + mean
        noise = np.around(noise, 0)
        
        return cv2.warpAffine(noise,self.M_marker_with_margin,(self.origin_image_shape[1],self.origin_image_shape[0]))
        

    def addMarker(self, origin_image, marker_pixel_pos=None, marker_yaw=None, maker_scale=None):
        
    
        if not self.initialized:
            self.origin_image_shape = origin_image.shape
            self.initialized = True
            
        if marker_pixel_pos is not None:
            # set Marker pixel position
            marker_pixel_pos_np = np.float32(marker_pixel_pos)
            assert marker_pixel_pos_np.shape == (2,1)
            self.transformMarkerImage(marker_pixel_pos, marker_yaw, maker_scale)

        
        noise = self.generateNoise()
        
        # correct the luminosity for Marker area 
        processed_image = (noise + self.marker_image_transformed) * self.marker_weight_trasnformed + origin_image * self.bg_weight
        return processed_image.astype(np.uint8)


if __name__ == "__main__":
    # example, add the Marker image to the observation image
    origin_image = cv2.imread("/home/gaspard/Documents/ros_omnirobot/robotics-rl-srl-omnirobot/data/omnirobot_real_20190125_155902/record_000/frame000001.jpg",cv2.IMREAD_COLOR)
    marker_image_with_margin = cv2.imread("/home/gaspard/Documents/ros_omnirobot/robotics-rl-srl-omnirobot/data/marker_640x640_1_margin_4.jpg")
    marker_render = MarkerRender(noise_var=1.0)
    marker_render.setMarkerImage(marker_image_with_margin, [4,4,4,4])

    result = marker_render.addMarker(origin_image, marker_pixel_pos=[50,50], marker_yaw=0) # row, col, angle(rad) 

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
