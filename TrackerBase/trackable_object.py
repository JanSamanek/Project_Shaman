from TrackerBase.kalman_filter import KalmanFilter


class TrackableObject:

    def __init__(self, center_x, center_y, ID):
        self.kf = KalmanFilter((center_x, center_y))
        self.centroid = center_x, center_y
        self.ID = ID
        self.disappeared_count = 0
        self.box = None
        self.predicted_centroid = None

    def predict(self):
        self.predicted_centroid = self.kf.predict()
        return self.predicted_centroid

    def apply_kf(self, meas):
        self.kf.update(meas)
        return self.centroid
