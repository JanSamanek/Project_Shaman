from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from trackable_object import TrackableObject


class PersonCenterTracker:

    def __init__(self, max_disappeared=50, max_distance=80):

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.nextID = 0
        self.to_dict = OrderedDict()
        self.disappeared_dict = OrderedDict()

    def _register(self, person_center):
        # assign new person center
        self.to_dict[self.nextID] = TrackableObject(*person_center, self.nextID)
        # reset disappeared countdown
        self.disappeared_dict[self.nextID] = 0
        # update ID
        self.nextID += 1

    def _deregister(self, ID):
        # delete person from register
        del self.to_dict[ID]
        del self.disappeared_dict[ID]

    def update(self, boxes):
        # if no boxes are present
        if len(boxes) == 0:
            # loop over the keys and mark persons as disappeared
            for ID in self.disappeared_dict.keys():
                self.disappeared_dict[ID] += 1
                # if person has disappeared more than the threshold is, delete them
                if self.disappeared_dict[ID] >= self.max_disappeared:
                    self._deregister(ID)

            return self.to_dict

        # if new boxes present initialize new_persons_centers
        new_persons_centers = np.zeros((len(boxes), 2))
        # calculates the centre and assigns in to the new_persons_center variable
        for row, (left, top, width, height) in enumerate(boxes):
            center = _calculate_center(left, top, width, height)
            new_persons_centers[row] = center

        # if we haven't registered any new persons yet, register the new persons centers
        if len(self.to_dict) == 0:
            for row in range(0, len(new_persons_centers)):
                self._register(new_persons_centers[row])

        # else calculate the distances between points and assign new coordinates to persons centers
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.to_dict.keys())
            # objectCentroids = [to.predict() for to in self.to_dict.values()]   # list(self.to_dict.values())
            for to in self.to_dict.values():
                to.predict()
            predicted_centroids = [to.predicted_centroid for to in self.to_dict.values()]
            #####xxxxxxxxxxxxxxxxx############

            D = dist.cdist(np.array(predicted_centroids), new_persons_centers)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                objectID = objectIDs[row]
                self.to_dict[objectID].centroid = new_persons_centers[col]
                self.disappeared_dict[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared_dict[objectID] += 1

                    if self.disappeared_dict[objectID] > self.max_disappeared:
                        self._deregister(objectID)
            else:
                for col in unusedCols:
                    self._register(new_persons_centers[col])

            # return the set of trackable objects
        return self.to_dict


def _calculate_center(start_x, start_y, end_x, end_y):
    center_x = (start_x + end_x) / 2.0
    center_y = (start_y + end_y) / 2.0
    return center_x, center_y
