import numpy as np 
import cv2
from tracker.tracker import Tracker
import time
#import imageio
import random
images = []

def createimage(w,h):
	size = (w, h, 1)
	img = np.ones((w,h,3),np.uint8)*255
	return img

def main():
	data = np.array(np.load('Detections.npy'))[0:15,0:300000,0:300000]
	tracker = Tracker(150, 30, 5)
	skip_frame_count = 0
	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(127, 127, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]

	out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (512,512))
	del_test = [5, 6]
	del_test_range = 60
	circles = [5,11]
	test_range = [20,40]
	for i in range(data.shape[1]): #data.shape[1]
		if (i >= test_range[0] and i < test_range[1]):
			centers = data[:,i,:]
			centers  = centers[circles[0]:circles[1],:]
			#centers = np.array([])
		else:
			centers = data[:,i,:]
		frame = createimage(512,512)
		predict_member = []
		#if (len(centers) > 0):
		
		if i == del_test_range:
			tracker.remove(del_test)
		tracker.predict() #Predict State
		for j in range(len(tracker.tracks)): #Get Length of member tracking | for j in range(len(tracker.tracks))
			if (len(tracker.tracks[j].trace) > 1): #Check trace queue of tracking object
				x = int(tracker.tracks[j].trace[-1][0,0]) #Get Predicted X from Kalman
				y = int(tracker.tracks[j].trace[-1][0,1]) #Get Predicted Y from Kalman
				tl = (x-10,y-10) # X min, Y min
				br = (x+10,y+10) # X max, Y max
				predict_member.append([x,y])
				cv2.rectangle(frame,tl,br,track_colors[j],1)
				cv2.putText(frame,"TrackID" + str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[j],2)
				"""
				for k in range(len(tracker.tracks[j].trace)):
					x = int(tracker.tracks[j].trace[k][0,0])
					y = int(tracker.tracks[j].trace[k][0,1])
					cv2.circle(frame,(x,y), 3, track_colors[j],-1)
				cv2.circle(frame,(x,y), 6, track_colors[j],-1)
				"""
			if len(centers) > 0:
				if i > test_range[0] and i < test_range[1]:
					for circle in range(circles[0],circles[1]):
						cv2.circle(frame,(int(data[circle ,i,0]),int(data[circle,i,1])), 6, (0,0,0),-1)
				else:
					cv2.circle(frame,(int(data[j ,i,0]),int(data[j,i,1])), 6, (0,0,0),-1)

		cv2.imshow('image',frame)
		tracker.update(centers) #Update State
		out.write(frame)
		# cv2.imwrite("image"+str(i)+".jpg", frame)
		# images.append(imageio.imread("image"+str(i)+".jpg"))
		time.sleep(0.1)
		#cv2.waitKey(0)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

	# imageio.mimsave('Multi-Object-Tracking.gif', images, duration=0.08)
			
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

if __name__ == '__main__':
	main()
