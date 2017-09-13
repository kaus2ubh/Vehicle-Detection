import cv2
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from functions.functions import *
from collections import deque
from skimage.feature import hog

#Pipeline class used for video detection pipeline
class Pipeline():
	def __init__(self):
		self.current_frame_number = 0
		self.svc_loaded = None
		self.color_space = 'YCrCb'
		self.spatial_size = (32,32)
		self.hist_bins = 32
		self.orient = 9
		self.pix_per_cell = 8
		self.cell_per_block = 2
		self.hog_channel = 'ALL'
		self.spatial_feat = True
		self.hist_feat = True
		self.hog_feat = True
		self.data_scaler = None
		self.heat_map_threshold = 4
		self.debug = 0
		#self.search_window_size = [(64,64,1),(96,96,1.5),(128,128,2),(160,160,2.5),(192,192,3)]
		#self.y_stop_coordinate =  [500,550,600,650,720]
		#self.ystartstop = [(320,500,1),(400,600,1.5),(400,650,2),(400,700,2.5)]
		#self.ystartstop = [(360, 560, 1.5), (400, 600, 1.8), (440, 700, 2.5)]
		##self.ystartstop = [(360, 560, 1), (400, 600, 1.5), (440, 700, 2)]
		self.ystartstop = [(360,460,1),(360,560,1.5),(400,600,2),(400,700,2.5)]
		#self.ystartstop = [(400, 500, 1), (400, 550, 1.5), (400, 600, 2), (400,650,2.5), (400,720,3)]
		self.bufferlen = 10
		self.search_offset = 50
		self.heat_map_buffer = deque(maxlen=self.bufferlen)
		self.bboxes=[]
		self.sliding_windows=[(64,500,1),(96,550,1.5),(128,600,2),(160,650,2.5),(192,700,3)]

	# Define a single function that can extract features using hog sub-sampling and make predictions
	# similar to function proviede in a quiz, but modified to meet video processing requirements
	def find_cars(self, img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
	    draw_img = np.copy(img)
	    img = img.astype(np.float32)/255
	    
	    #img_tosearch = img[ystart:ystop,600:1280,:]
	    #search only right half of the image
	    #xstart=600
	    img_tosearch = img[ystart:ystop,xstart:xstop,:]
	    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
	    if scale != 1:
	        imshape = ctrans_tosearch.shape
	        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
	        
	    ch1 = ctrans_tosearch[:,:,0]
	    ch2 = ctrans_tosearch[:,:,1]
	    ch3 = ctrans_tosearch[:,:,2]

	    # Define blocks and steps as above
	    nxblocks = (ch1.shape[1] // pix_per_cell)-1
	    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
	    nfeat_per_block = orient*cell_per_block**2
	    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	    window = 64
	    nblocks_per_window = (window // pix_per_cell)-1 
	    cells_per_step = 2  # Instead of overlap, define how many cells to step
	    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
	    
	    # Compute individual channel HOG features for the entire image
	    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	    
	    detected_boxes = []
	    for xb in range(nxsteps):
	        for yb in range(nysteps):
	            ypos = yb*cells_per_step
	            xpos = xb*cells_per_step
	            # Extract HOG for this patch
	            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
	            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

	            xleft = xpos*pix_per_cell
	            ytop = ypos*pix_per_cell

	            # Extract the image patch
	            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
	          
	            # Get color features
	            spatial_features = bin_spatial(subimg, size=spatial_size)
	            hist_features = color_hist(subimg, nbins=hist_bins)

	            # Scale features and make a prediction
	            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
	            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
	            test_prediction = svc.predict(test_features)
	            
	            if test_prediction == 1:
	                xbox_left = np.int(xleft*scale)
	                ytop_draw = np.int(ytop*scale)
	                win_draw = np.int(window*scale)
	                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
	                detected_boxes.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))
	    return detected_boxes

	def Pipeline(self, img):
		self.current_frame_number = self.current_frame_number+1
		bxstart=0
		bxstop=1280
		bystart=0
		bystop=720
		if(self.current_frame_number%24==0 or self.current_frame_number==1):
			#print(self.current_frame_number,bystart,bystop)
			bxstart=600
			bxstop=1280
			bystart=0
			bystop=720
		elif((len(self.bboxes))):
			#print("num of bboxes=",len(self.bboxes))
			bxstart=min(self.bboxes,key=lambda t: t[0][0])[0][0]-50
			bxstop =max(self.bboxes,key=lambda t: t[1][0])[1][0]+50
			bystart=min(self.bboxes,key=lambda t: t[0][1])[0][1]-50
			bystop =max(self.bboxes,key=lambda t: t[1][1])[1][1]+50

		detected_boxes = []
		for (ystart,ystop,scale) in self.ystartstop:
			if(bystop<bystart or bxstop<bxstart):
				##print("skip detection")
				continue
			ystart=max(bystart,ystart)
			ystop =min(bystop,ystop)
			xstart=max(bxstart,600)
			xstop =min(bxstop,1280)
			#print(self.current_frame_number,bystop,bystart)
			detected_boxes = detected_boxes + self.find_cars(img, ystart, ystop, xstart, xstop, scale, self.svc_loaded, self.data_scaler, self.orient, 
				self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)
		#print(detected_boxes)
		heat_zero = np.zeros_like(img[:,:,0]).astype(np.float)
		heat_map = add_heat(heat_zero, detected_boxes)
		heat_thresholded = apply_threshold(heat_map, self.heat_map_threshold)
		self.heat_map_buffer.append(heat_thresholded)

		#detected blob must be in min 7 frames consecutively to be detected
		#used to weed out false positives
		if(len(self.heat_map_buffer)==self.heat_map_buffer.maxlen or self.debug==1):
			if(self.debug!=1):
				#print("asa")
				heat_sum = np.sum(np.array(self.heat_map_buffer), axis=0)
				heat_final = apply_threshold(heat_sum, self.heat_map_buffer.maxlen-2)
				labels = label(heat_final)
				#labels = label(heat_thresholded)
				img_final, self.bboxes = draw_labeled_bboxes(img, labels)
				return img_final
			else:
				labels = label(heat_thresholded)
				img_final, temp = draw_labeled_bboxes(img, labels)
				hot_boxes = draw_boxes(img, detected_boxes, color=(0, 0, 255), thick=6)

				bxstart=min(detected_boxes,key=lambda t: t[0][0])[0][0]-50
				bxstop =max(detected_boxes,key=lambda t: t[1][0])[1][0]+50
				bystart=min(detected_boxes,key=lambda t: t[0][1])[0][1]-50
				bystop =max(detected_boxes,key=lambda t: t[1][1])[1][1]+50
				#print(detected_boxes)

				restricted_box = [((bxstart,bystart),(bxstop,bystop))]
				print(restricted_box)
				restricted_space = draw_boxes(img, restricted_box, color=(255, 0, 0), thick=6)
				return img_final, heat_map, hot_boxes, restricted_space
		else:
			#print("asa11")
			return img

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

#conversion to YCrCb done before binning itself
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    
	#bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    #bins_range = (0, 256)
    #channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    #channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    #channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes=[]
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img, bboxes

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
    	hog_features = []
    	for channel in range(feature_image.shape[2]):
    		#print(channel, feature_image[:,:,channel].shape, orient, type(orient), pix_per_cell, cell_per_block)
    		hog_features.extend(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
    	img_features.append(hog_features)
    return np.concatenate(img_features)