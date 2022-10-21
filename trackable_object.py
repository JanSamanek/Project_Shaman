from kalman_filter import KalmanFilter


class TrackableObject:

    def __init__(self, center_x, center_y, ID):
        self.kf = KalmanFilter()
        self.centroid = center_x, center_y
        self.ID = ID
        self.disappeared_count = 0
        self.predicted_centroid = self.kf.predict(*self.centroid)

    def predict(self):
        self.predicted_centroid = self.kf.predict(*self.centroid)
        return self.predicted_centroid
