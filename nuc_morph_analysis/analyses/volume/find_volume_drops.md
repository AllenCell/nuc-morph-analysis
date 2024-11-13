# workflow

0. inputs are volume trajectory and power law fit (used for detrending)
0. both inputs are interpolated so there is a value at each point in time (necessary for smoothing)
1. volume trajectory is smoothed (--> smoothed volume trajectory)
2. smoothed volume trajectory is detrended by subtracting the power law fit (--> detrended smoothed volume trajectory)
3. search for volume dips by putting inverse of detrended smoothed volume trajectory into scipy.signal.find_peaks
4. peaks are removed from `left_base` to `right_base` by linear interpolation or filling with nans. 
5.