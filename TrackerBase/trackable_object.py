from TrackerBase.kalman_filter import KalmanFilter


class TrackableObject:

    def __init__(self, center_x, center_y, ID):
        self.kf = KalmanFilter((center_x, center_y))
        self.centroid = center_x, center_y
        self.ID = ID
        self.disappeared_count = 0
        self.box = None
        self.predicted_centroid = None
        self.measured_centroid = None
        
    def predict(self, camera_rotation):
        self.predicted_centroid = self.kf.predict(camera_rotation)
        return self.predicted_centroid

    def apply_kf(self, meas):
        return self.kf.update(meas)
