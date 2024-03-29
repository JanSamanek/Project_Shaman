from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from TrackerBase.trackable_object import TrackableObject
import json


class PersonTracker:

    def __init__(self, center_to, max_disappeared=50, max_distance=200):
        print("[INF] Initalizing tracker base...")
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.nextID = 0
        self.to_dict = OrderedDict()
        self._register(center_to)

        with open("settings.json") as json_file:
            data = json.load(json_file)
            self.FoV = data['FoV']
            self.dt = data['dt']
            self.img_width = data['img_width']

    def _register(self, person_center):
        # assign new person center
        self.to_dict[self.nextID] = TrackableObject(*person_center, self.nextID)
        # update ID
        self.nextID += 1

    def _deregister(self, ID):
        # delete person from register
        del self.to_dict[ID]

    def update(self, boxes, camera_rotation=0):
        # if no boxes are present
        if len(boxes) == 0:
            IDs_to_delete = []
            # loop over the keys and mark persons as disappeared
            for to in self.to_dict.values():
                to.disappeared_count += 1
                to.centroid = None
                to.box = None
                to.measured_centroid = None
                # if person has disappeared more than the threshold is, delete them
                if to.disappeared_count >= self.max_disappeared:
                    IDs_to_delete.append(to.ID)

            for ID in IDs_to_delete:
                self._deregister(ID)

            return self.to_dict

        # if new boxes present initialize new_centers
        new_centers = np.zeros((len(boxes), 2))
        center_box_dict = {}
        # calculates the centre and assigns in to the new_persons_center variable
        for i, box in enumerate(boxes):
            centroid = _calculate_center(*box)
            new_centers[i] = centroid
            center_box_dict[tuple(centroid)] = box

        # if we haven't registered any new persons yet, register the new persons centers
        if len(self.to_dict) == 0:
            for i in range(0, len(new_centers)):
                self._register(new_centers[i])
        # else calculate the distances between points and assign new coordinates to persons centers
        else:
            # grab the set of object IDs and corresponding centroids
            IDs = [to.ID for to in self.to_dict.values()]
            predicted_centroids = [to.predict(camera_rotation) for to in self.to_dict.values()]

            D = dist.cdist(np.array(predicted_centroids), new_centers)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):

                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                ID = IDs[row]
                to = self.to_dict[ID]
                new_center = new_centers[col]

                # assign box
                ################################################################
                to.box = center_box_dict[tuple(new_center)]
                to.measured_centroid = new_center
                to.disappeared_count = 0
                
                robot_pixel_movement = int(camera_rotation*self.dt*self.img_width/((self.FoV*np.pi*2)/360))
                
                to.centroid = to.apply_kf(new_center) if abs(robot_pixel_movement) < 10 else None
                ##################################################################

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unused_rows:
                    ID = IDs[row]
                    to = self.to_dict[ID]
                    to.disappeared_count += 1
                    to.centroid = None    
                    to.measured_centroid = None
                    to.box = None

                    if to.disappeared_count > self.max_disappeared:
                        self._deregister(ID)
            else:
                for col in unused_cols:
                    self._register(new_centers[col])

        return self.to_dict


def _calculate_center(start_x, start_y, end_x, end_y):
    center_x = int((start_x + end_x) / 2.0)
    center_y = int((start_y + end_y) / 2.0)
    return center_x, center_y
