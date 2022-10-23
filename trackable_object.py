from kalman_filter import KalmanFilter


class TrackableObject:

    def __init__(self, center_x, center_y, ID):
        self.kf = KalmanFilter((center_x, center_y))
        self.centroid = center_x, center_y
        self.box = None  # TODO ?
        self.ID = ID
        self.disappeared_count = 0
        self.predicted_centroid = None   # TODO ?

    def predict(self):
        if self.disappeared_count != 0:
            self.predicted_centroid = self.kf.predict(*self.predicted_centroid)
        else:
            self.predicted_centroid = self.kf.predict(*self.centroid)

        return self.predicted_centroid
