import numpy as np 
from tracker.kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque


class Tracks(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId):
		super(Tracks, self).__init__()
		self.KF = KalmanFilter() #Init all A,B,
		self.KF.predict()
		self.KF.correct(np.matrix(detection).reshape(2,1))
		self.prediction = detection.reshape(1,2)
		self.detection = detection.reshape(1,2)
		self.trackId = trackId
		self.skipped_frames = 0
		self.step = 0

	def predict(self):
		self.prediction = np.array(self.KF.predict()).reshape(1,2)
		
	def update(self,detection):
		self.KF.correct(np.matrix(detection).reshape(2,1))


class Tracker(object):
	"""docstring for Tracker"""
	def __init__(self, dist_threshold=150, max_frame_skipped=30, minRoi=0, maxRoi=0, riseTime=20):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.trackId = 0
		self.tracks = []
		self.previous_index_tracks = 0
		self.riseTime = riseTime
	
	def remove(self, del_tracks):
		del_tracks.sort(reverse=True) # To Remove from last to front because if we dont use reverse it will cause out of length
		for i in del_tracks:
			del self.tracks[i] # Assume self.tracks = [track0, track1, track2, track3, track4, track5] del self.track[0] @ i=1 ->  [track1, track2, track3, track4, track5] 
		self.previous_index_tracks = len(self.tracks) # Update previous_index_tracks

	def predict(self):
		if len(self.tracks) != 0:
			for i in range(len(self.tracks)):
				self.tracks[i].predict() #Update predict value Xp(PredictedState), Pp(errorCovraiance)
				self.tracks[i].detection = self.tracks[i].prediction
	
	def update(self, detections):
	
		if (detections.shape[0] >= self.previous_index_tracks) and (detections.shape[0] != 0): #Check if index of tracks are greater or equal than previous one
			if len(self.tracks) == 0: #Check if first time running
				for i in range(detections.shape[0]): #Loop to get all detections
					track = Tracks(detections[i], self.trackId) #Init Tracks from 0 to n
					self.tracks.append(track) #Add init track to self.tracks
					self.trackId += 1 #Update trackID for preventing same ID used

			N = len(self.tracks) #Length of self.tracks
			M = len(detections) #Length of detections
			cost = [] # Cost
			""" Calculate Minimum Assignments """
			for i in range(N):
				diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1) # Use Normalization 2 for compare Prediction@[k-1] and Real
				cost.append(diff)
			""" ############################# """
			cost = np.array(cost)
			row, col = linear_sum_assignment(cost) # Minimum Assignment row = [0,1,2,3,n], col = [ass0,ass1,ass2,assn]
			assignment = [-1]*N # Initial null assignments
			for i in range(len(row)):
				assignment[row[i]] = col[i] # Assigned all track


			un_assigned_tracks = []

			#Check RedZone


			for i in range(len(assignment)): #Loop for finding un_assignments
				if assignment[i] != -1: # Check If assignments has been assigned
					if (cost[i][assignment[i]] > self.dist_threshold) and (self.tracks[i].step > self.riseTime): # If assigned track is exceeding to distance threshold, It will be declared to unassigned track
						assignment[i] = -1 # Assign which track to -1
						un_assigned_tracks.append(i) # Append to un_assigned_tracks
					else:
						self.tracks[i].skipped_frames +=1 #If frame has been skipped


			del_tracks = []
			for i in range(len(self.tracks)):
				if self.tracks[i].skipped_frames > self.max_frame_skipped:
					del_tracks.append(i)

			if len(del_tracks) > 0:
				for i in range(len(del_tracks)):
					del self.tracks[i]
					del assignment[i]

			#If new track has been added more please uncomment this
			for i in range(len(detections)): 
				if i not in assignment:
					track = Tracks(detections[i], self.trackId)
					self.tracks.append(track)
					self.trackId +=1



			for i in range(len(assignment)):
				if(assignment[i] != -1):
					self.tracks[i].skipped_frames = 0
					self.tracks[i].detection = detections[assignment[i]]
					self.tracks[i].update(detections[assignment[i]]) # Update State

			""" Update Previous equals current index tracks """
			self.previous_index_tracks = detections.shape[0]

		elif (detections.shape[0] < self.previous_index_tracks) : #Check if index of tracks are greater or equal than previous one
			if detections.shape[0] > 0:
				N = len(self.tracks)
				M = len(detections)
				cost = []
				preds_all = []
				for i in range(N):
					preds_all.append(self.tracks[i].prediction[0])

				for i in range(M):
					diff = np.linalg.norm(preds_all - detections[i], axis=1) # Use Normalization 2 for compare Prediction@[k-1] and Real
					cost.append(diff)
				cost = np.array(cost)*0.1
				row, col = linear_sum_assignment(cost) # Minimum Assignment
				assignment = [-1] * len(col)
				i = 0

				un_assigned_tracks = []

				while (len(assignment) < N ) or (-1 in assignment):
					if i not in col:
						col = np.append(col, i)
						assignment.append(i)
						detections = np.append(detections, [self.tracks[i].detection[0,0], self.tracks[i].detection[0,1]])
						un_assigned_tracks.append(i)
					else:
						assignment[i] = col[i]
						i+=1
				
				detections = detections.reshape(len(self.tracks), 2) # [0.1,0.1,0.2,0.2,0.3,0.3] -> [[0.1 0.1],[0.2 0.2],[0.3 0.3]]


				""" Assigns Tracks """
				for det_index, i in enumerate(assignment):
					if(assignment[i] != -1):
						#self.tracks[i].skipped_frames = 0
						self.tracks[i].detection = detections[assignment[i]]
						self.tracks[i].update(detections[det_index]) # Update State

				""" Delete Tracks """
				del_tracks = []
				for i in un_assigned_tracks:
					self.tracks[i].skipped_frames +=1
					if self.tracks[i].skipped_frames > self.max_frame_skipped :
						del_tracks.append(i)

				if len(del_tracks) > 0:
					del_tracks.sort(reverse=True) # To Remove from last to front because if we dont use reverse it will cause out of length
					self.previous_index_tracks = M # Update previous_index_tracks
					for i in del_tracks:
						del self.tracks[i] # Assume self.tracks = [track0, track1, track2, track3, track4, track5] del self.track[0] @ i=1 ->  [track1, track2, track3, track4, track5] 
						del assignment[i] # Assume self.tracks = [track0, track1, track2, track3, track4, track5] del self.track[0] @ i=1 ->  [track1, track2, track3, track4, track5]




			elif (detections.shape[0] == 0) and len(self.tracks) > 0:
				""" Predict Assigned """
				for j in range(len(self.tracks)):
					detections = np.append(detections , [self.tracks[j].detection[0,0], self.tracks[j].detection[0,1]])
					detections = detections.reshape(j+1, 2) # [0.1,0.1,0.2,0.2,0.3,0.3] -> [[0.1 0.1],[0.2 0.2],[0.3 0.3]]
				N = len(self.tracks)
				M = len(detections)
				cost = []
				for i in range(N):
					diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1) # Use Normalization 2 for compare Prediction@[k-1] and Real
					cost.append(diff)

				cost = np.array(cost)*0.1
				row, col = linear_sum_assignment(cost) # Minimum Assignment
				assignment = [-1]*N # Initial null assignments
				for i in range(len(row)):
					assignment[row[i]] = col[i] # Assigned all track

				un_assigned_tracks = []

				for i in range(len(assignment)): #Loop for finding un_assignments
					self.tracks[i].skipped_frames +=1
					if assignment[i] != -1: # Check If assignments has been assigned
						if (cost[i][assignment[i]] > self.dist_threshold): # If assigned track is exceeding to distance threshold, It will be declared to unassigned track
							assignment[i] = -1 # Assign which track to -1
							un_assigned_tracks.append(i) # Append to un_assigned_tracks

				
				for det_index, i in enumerate(assignment):
					if(assignment[i] != -1):
						#self.tracks[i].skipped_frames = 0
						self.tracks[i].update(detections[det_index]) # Update State
						self.tracks[i].detection = detections[assignment[i]]


				del_tracks = []
				for i in range(len(self.tracks)):
					if self.tracks[i].skipped_frames > self.max_frame_skipped :
						del_tracks.append(i)

				""" Delete All tracks """
				if len(del_tracks) > 0:
					del_tracks.sort(reverse=True) # To Remove from last to front because if we dont use reverse it will cause out of length
					self.previous_index_tracks = 0 # Update previous_index_tracks
					for i in del_tracks:
						del self.tracks[i] # Assume self.tracks = [track0, track1, track2, track3, track4, track5] del self.track[0] @ i=1 ->  [track1, track2, track3, track4, track5] 
						del assignment[i] # Assume self.tracks = [track0, track1, track2, track3, track4, track5] del self.track[0] @ i=1 ->  [track1, track2, track3, track4, track5]

				
		for ind, track in enumerate(self.tracks):
			track.step += 1 # Update Track Steps for each track
