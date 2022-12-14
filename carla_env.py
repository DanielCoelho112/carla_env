import glob
from msilib.schema import Class
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
import math

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10 

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None 
    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]
        
    def reset(self):
        self.collision_hist = [] # the collision events returns a list of the collisions.
        self.actor_list = [] # we always need to track actors
        
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        
        # here it is like gazebo. When we spawn an actor, the actor falls from the sky, and sometimes it takes some time to produce the sensor data.
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4) # to let everyting settle down.
        
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        while self.front_camera is None:
            time.sleep(0.01)
        
        self.episode_start = time.time() # episodes are 10 seconds fixed
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)
        
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:,:,:3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3 # this attribute will contain the last image.
    
    def step(self, action):
        if action == 0: # left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        elif action == 1: # straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0 * self.STEER_AMT))
        elif action == 2: # right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))
        
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        if len(self.collision_hist)!= 0: # if any collision, abort episode
            done = True
        
        elif kmh < 50:
            done = False 
            reward = -1
        
        else:
            done = False 
            reward = 1
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True 
        
        return self.front_camera, reward, done, None