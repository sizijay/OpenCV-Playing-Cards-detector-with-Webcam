from threading import Thread
import cv2

class VideoStream:
    """Camera object"""
    def __init__(self, resolution=(640,480),framerate=30,PiOrUSB=1,src=0):

        self.PiOrUSB = PiOrUSB
        self.stream = cv2.VideoCapture(src)
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()              
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):	
        return self.frame

    def stop(self):	
        self.stopped = True
