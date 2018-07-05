import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10


frameRateCalc = 1
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_DUPLEX

videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
                          
time.sleep(1)

path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Cards/')
train_suits = Cards.load_suits( path + '/Cards/')

camQuit = False 

while not camQuit :    
    img = videostream.read()   
    t1 = cv2.getTickCount()   
    prePro = Cards.preprocess_image(img)   
    cntSort, cntIsCard = Cards.find_cards(prePro)
  
    if len(cntSort) != 0:        
        cards = []
        k = 0        
        for i in xrange(len(cntSort)):
            if (cntIsCard[i] == 1):             
                cards.append(Cards.preprocess_card(cntSort[i],img))
                cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits) 
                img = Cards.draw_results(img, cards[k])
                k = k + 1	    
        
        if (len(cards) != 0):
            tempCnts = []
            for i in xrange(len(cards)):
                tempCnts.append(cards[i].contour)
            cv2.drawContours(img,tempCnts, -1, (255,0,0), 2)     
    
    cv2.putText(img,"FPS: "+str(int(frameRateCalc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)    
    cv2.imshow("Playing Cards Detector",img)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frameRateCalc = 1/time1
      
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        camQuit = True
        

cv2.destroyAllWindows()
videostream.stop()
os._exit(0)

