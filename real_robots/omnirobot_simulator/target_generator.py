import cv2
import numpy as np
from matplotlib import pyplot as plt
class TargetGenerator():
    def __init__(self, noise_var):
        self.target_image_with_margin = None # need to read from file
        self.margin = [0.0,0.0,0.0,0.0]
        
        self.origin_image_shape = None
        self.target_image_transformed = None
        self.target_weight_trasnformed = None
        self.bg_weight = None # back ground mask, inverse of target_mask

        self.noise_var = noise_var
        
        self.initialized = False
        
    def setTargetImage(self,target_image_with_margin, margin):
        self.target_image_with_margin = target_image_with_margin
        self.margin = margin
        
        
    def transformTargetImage(self, target_pixel_pos):
        self.target_pixel_pos = target_pixel_pos
        self.M_target_with_margin =  cv2.getRotationMatrix2D((self.target_image_with_margin.shape[1]/2,\
                                                              self.target_image_with_margin.shape[0]/2),\
                                                             self.target_pixel_pos[2]*180/np.pi,1)
        self.M_target_with_margin[0,2] += self.target_pixel_pos[1] - self.target_image_with_margin.shape[1]/2
        self.M_target_with_margin[1,2] += self.target_pixel_pos[0] -  self.target_image_with_margin.shape[0]/2

        # setup margin's weight
        target_weight = np.ones(self.target_image_with_margin.shape[0:3], np.float32)
        for i in range(self.margin[0]):
            target_weight[i,:] = i/self.margin[0]
        for i in range(self.margin[1]):
            target_weight[:,i] = i/self.margin[1]
        for i in range(self.margin[2]):
            target_weight[-i-1,:] = i/self.margin[2]
        for i in range(self.margin[3]):
            target_weight[:,-i-1] = i/self.margin[3]
            
        self.target_image_transformed = cv2.warpAffine(self.target_image_with_margin,self.M_target_with_margin,\
                                                       (self.origin_image_shape[1],self.origin_image_shape[0])) 
        
        self.target_weight_trasnformed = cv2.warpAffine(target_weight,self.M_target_with_margin,\
                                                      (self.origin_image_shape[1],self.origin_image_shape[0]))  # white: target part
        
        self.target_image_transformed_LAB = cv2.cvtColor(self.target_image_transformed, cv2.COLOR_BGR2LAB)

        self.bg_weight = 1.0 - self.target_weight_trasnformed # white: origin image part

    def generateNoise(self):
        mean = 0.0
        noise = np.random.standard_normal(self.target_image_with_margin.shape) * self.noise_var + mean
        noise = np.around(noise, 0)
        
        return cv2.warpAffine(noise,self.M_target_with_margin,(self.origin_image_shape[1],self.origin_image_shape[0]))
        

    def addTarget(self, origin_image, target_pixel_pos=None):
        
    
        if not self.initialized:
            self.origin_image_shape = origin_image.shape
            self.initialized = True
            
        if target_pixel_pos is not None:
            # set target pixel position
            self.transformTargetImage(target_pixel_pos)
    
        noise = self.generateNoise()
        
        # correct the luminosity for target area 
        origin_image_LAB = cv2.cvtColor(origin_image, cv2.COLOR_BGR2LAB)
        target_image_transformed_corrected = self.target_image_transformed_LAB
        target_image_transformed_corrected[0] = origin_image_LAB[0]
        target_image_transformed_corrected = cv2.cvtColor(target_image_transformed_corrected, cv2.COLOR_LAB2BGR)
        processed_image = (noise + target_image_transformed_corrected) * self.target_weight_trasnformed + origin_image * self.bg_weight
        return processed_image.astype(np.uint8)


if __name__ == "__main__":
    # example, add the target image to the observation image
    origin_image = cv2.imread("/home/gaspard/Documents/ros_omnirobot/robotics-rl-srl-omnirobot/data/omnirobot_real_20190125_155902/record_000/frame000001.jpg",cv2.IMREAD_COLOR)
    target_image_with_margin = cv2.imread("/home/gaspard/Documents/ros_omnirobot/robotics-rl-srl-omnirobot/data/target_640x640_1_margin_4.jpg")
    target_generator = TargetGenerator(noise_var=1.0)
    target_generator.setTargetImage(target_image_with_margin, [4,4,4,4])

    result = target_generator.addTarget(origin_image, target_pixel_pos=[50,50,0]) # row, col, angle(rad) 

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
