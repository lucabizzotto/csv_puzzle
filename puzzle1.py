"""
it create random puzzle characterized by 4 pieces where each piece is compose by 4 vertex
"""

import numpy as np
import random as rnd
import cv2
import torch
import Modules

class Puzzle:

    def __init__(self, device, canvas_height=600, canvas_width=600, n_lines=2, new = True):
        if new:
            self.canvas_width = canvas_width
            self.canvas_height = canvas_height
            self.number_of_pieces, self.centroids, self.polygon_points = self.create(n_lines)
            self.piece_color = [(rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)) for _ in
                                range(self.number_of_pieces)]
            # from the centroid we can have the all the other points of the polygon
            self.difference = self.find_difference()
            # save index of shared vertex of the puzzle pieces
            self.index_shared_vertex = self.shared_vertex()
            # difference of normalize coordinates
            self.difference_norm = self.normalize_diff(device)
        # we want to load values of puzzle already created
        else:
            self.canvas_width = None
            self.canvas_height = None
            self.number_of_pieces, self.centroids, self.polygon_points = None, None, None
            self.piece_color = None
            # from the centroid we can have the all the other points of the polygon
            self.difference = None
            # save index of shared vertex of the puzzle pieces
            self.index_shared_vertex = None
            self.difference_norm = None


    def set_field(self, centroids, polygon_points, piece_color, device,  canvas_height = 600, canvas_width = 600, number_of_pieces = 4):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.centroids = centroids
        self.polygon_points = polygon_points
        self.piece_color = piece_color
        # From the centroid we can have the all the other points of the polygon
        self.difference = self.find_difference()
        self.index_shared_vertex = self.shared_vertex()
        self.number_of_pieces = number_of_pieces
        # difference of normalize coordinates
        self.difference_norm = self.normalize_diff(self, device)




    def find_difference(self):
        """
        calculate the distance between the center and its vertex
        :return:
        """
        pp = torch.from_numpy(self.polygon_points.squeeze())
        center = self.centroids
        differences = torch.empty((4, 4, 2))
        for enum, i in enumerate(pp):
            diff = i - center[enum]
            differences[enum] = diff
        return differences

    def normalize_diff(self, device):
        """
        calcaulte the distance betwen the vertex and its center in normalize cordinates
        :param device:
        :return:
        """
        # normalize center
        centers_normalize = Modules.utility1.Utility.normalize_coordinate(self.centroids, self.get_height(), self.get_width(),
                                                                          device)
        # normalize polygon points
        polygon_points_normalize = Modules.utility1.Utility.normalize_coordinate(self.polygon_points, self.get_height(),
                                                                                 self.get_width(), device).squeeze()
        # find the difference between each vertex with its corresponding centroids
        differences = torch.empty((4, 4, 2))
        for enum, i in enumerate(polygon_points_normalize):
            diff = i - centers_normalize[enum]
            differences[enum] = diff
        return differences




    def create(self, n_lines):
        n_lines = n_lines // 2
        flag = False
        while not flag:
            # create  canvas
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
            flag = True
            # vertical line
            for i in range(n_lines):
                # Generate random starting and ending points for the line
                start_x = np.random.randint(0, self.canvas_width)
                end_x = np.random.randint(0, self.canvas_width)
                start_y = 0
                end_y = self.canvas_height - 1
                line1 = np.array([start_x, start_y, end_x, end_y])
                cv2.line(canvas, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            # horizontal line
            for i in range(n_lines):
                a_start_y = np.random.randint(0, self.canvas_height)
                a_end_y = np.random.randint(0, self.canvas_height)
                a_start_x = 0
                a_end_x = self.canvas_width - 1
                cv2.line(canvas, (a_start_x, a_start_y), (a_end_x, a_end_y), (0, 0, 255), 2)

                # represent line
                line2 = np.array([a_start_x, a_start_y, a_end_x, a_end_y])

            # fill the polygon created by line intersection
            # work with gray image
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            centroids = []
            polygon_points = []
            for i in range(len(contours)):
                # it aim to simplify the polyline by reducing the number of vertices based on epsilon value
                approx = cv2.approxPolyDP(contours[i], 0.015 * cv2.arcLength(contours[i], True), True)
                # only polygon with 4 vertex admit
                if len(approx) != 4:
                    flag = False

                polygon_points.append(approx)
                # find center of mass of polygon
                m = cv2.moments(approx)
                # m00 area of the polygon
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    centroids.append([cx, cy])
                else:
                    pass

        polygon_points = np.array(polygon_points)
        polygon_points = self.align_polygon_edges(polygon_points)
        # number of polygon created, centroid of each polygon, polygon points
        return len(contours), np.array(centroids), polygon_points

    def get_normalize_diff(self):
        return self.difference_norm

    def get_number_puzzle_pieces(self):
        return self.number_of_pieces

    def get_index_shared_vertex(self):
        return self.index_shared_vertex

    def get_piece_number(self):
        return self.number_of_pieces

    def get_centroids(self):
        return torch.from_numpy(self.centroids).to(torch.float32)

    def get_polygon_points(self):
        return torch.from_numpy(self.polygon_points)

    def get_puzzle_dimension(self):
        return self.canvas_height, self.canvas_width

    def get_puzzle(self):
        # create the puzzle
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        for enum, c in enumerate(self.polygon_points):
            canvas = cv2.fillPoly(canvas, [c], color=self.piece_color[enum])
        return canvas

    def get_piece_color(self):
        canvas = self.get_puzzle()
        return np.stack([canvas[int(i[1]), int(i[0])] for i in self.centroids])

    def get_height(self):
        return self.canvas_height

    def get_width(self):
        return self.canvas_width

    def get_difference(self):
        return self.difference



    @staticmethod
    def align_polygon_edges(polygons_point):
        original_shape = polygons_point.shape
        # reshape
        points = polygons_point.reshape((-1, 2))

        # Define the predefined distance threshold
        threshold_distance = 25.

        # Compute the pairwise differences between points
        differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # Compute the squared distances along the last axis and sum them
        squared_distances = np.sum(differences ** 2, axis=-1)
        # Compute the distance matrix
        distance_matrix = np.sqrt(squared_distances)
        # i want exclude diagonal element
        np.fill_diagonal(distance_matrix, np.inf)

        # Create a mask of points within the predefined distance threshold
        mask = distance_matrix < threshold_distance

        # Get the indices of true values in each row
        # position indicate the node the value is a list containing the close point to it
        indices = [np.where(row)[0] for row in mask]

        # To ensure that multiple close points to converge on a single point
        # i <- node
        for i in range(len(indices)):
            # more then one neighbor
            if indices[i].shape[0] > 1:
                # for each neighbor
                for n in indices[i]:
                    indices[n] = np.array([i])
                # choose my self
                indices[i] = np.array([i])
            # modify point following the choice
            if indices[i].shape[0] != 0:
                points[i] = points[indices[i][0]]
        # reshape to its original form
        points = points.reshape(original_shape)
        return points

    def shared_vertex(self):
        """

        :return: the index of vertex that share same coordinate
        """

        vertices = torch.from_numpy(self.polygon_points)
        # get the set of vertex
        vertex_set = torch.unique(vertices.reshape(-1, 2), dim=0)
        index_shared_vertex = []
        for v in vertex_set:
            # detect the same vertex
            logic = vertices == v
            # get the index
            indices = torch.nonzero(torch.all(logic == torch.tensor([1, 1], dtype=torch.uint8), dim=-1))
            if indices.shape[0] > 1:
                index_shared_vertex.append(indices)
        return index_shared_vertex

    def create_similar(self, n_lines):
        n_lines = n_lines // 2
        flag = False
        while not flag:
            # create  canvas
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
            flag = True
            # vertical line
            for i in range(n_lines):
                # Generate random starting and ending points for the line
                half_w = self.canvas_width // 2
                percentage = 0.2
                start_x = np.random.randint(half_w - int(self.canvas_width * percentage) , half_w + int(self.canvas_width * percentage) )
                end_x =  np.random.randint(half_w - int(self.canvas_width * percentage) , half_w + int(self.canvas_width * percentage) )
                start_y = 0
                end_y = self.canvas_height - 1
                line1 = np.array([start_x, start_y, end_x, end_y])
                cv2.line(canvas, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            # horizontal line
            for i in range(n_lines):
                half_h = self.canvas_height // 2
                percentage = 0.2
                a_start_y = np.random.randint(half_h - int(self.canvas_height * percentage) , half_h + int(self.canvas_height * percentage) )
                a_end_y =np.random.randint(half_h - int(self.canvas_height * percentage) , half_h + int(self.canvas_height * percentage) )
                a_start_x = 0
                a_end_x = self.canvas_width - 1
                cv2.line(canvas, (a_start_x, a_start_y), (a_end_x, a_end_y), (0, 0, 255), 2)

                # represent line
                line2 = np.array([a_start_x, a_start_y, a_end_x, a_end_y])

            # fill the polygon created by line intersection
            # work with gray image
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            centroids = []
            polygon_points = []
            for i in range(len(contours)):
                # it aim to simplify the polyline by reducing the number of vertices based on epsilon value
                approx = cv2.approxPolyDP(contours[i], 0.015 * cv2.arcLength(contours[i], True), True)
                # only polygon with 4 vertex admit
                if len(approx) != 4:
                    flag = False

                polygon_points.append(approx)
                # find center of mass of polygon
                m = cv2.moments(approx)
                # m00 area of the polygon
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    centroids.append([cx, cy])
                else:
                    pass

        polygon_points = np.array(polygon_points)
        polygon_points = self.align_polygon_edges(polygon_points)
        # number of polygon created, centroid of each polygon, polygon points
        return len(contours), np.array(centroids), polygon_points

    def create_rectangol_piece(self, n_lines):
        n_lines = n_lines // 2
        flag = False
        while not flag:
            # create  canvas
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
            flag = True
            # vertical line
            for i in range(n_lines):
                # Generate random starting and ending points for the line
                half_w = self.canvas_width // 2
                percentage = 0.15
                start_x = np.random.randint(half_w - int(self.canvas_width * percentage),
                                            half_w + int(self.canvas_width * percentage))
                end_x = np.random.randint(half_w - int(self.canvas_width * percentage),
                                          half_w + int(self.canvas_width * percentage))
                start_y = 0
                end_y = self.canvas_height - 1
                line1 = np.array([start_x, start_y, start_x, end_y])
                cv2.line(canvas, (start_x, start_y), (start_x, end_y), (0, 0, 255), 2)
            # horizontal line
            for i in range(n_lines):
                half_h = self.canvas_height // 2
                percentage = 0.15
                a_start_y = np.random.randint(half_h - int(self.canvas_height * percentage),
                                              half_h + int(self.canvas_height * percentage))
                a_end_y = np.random.randint(half_h - int(self.canvas_height * percentage),
                                            half_h + int(self.canvas_height * percentage))
                a_start_x = 0
                a_end_x = self.canvas_width - 1
                cv2.line(canvas, (a_start_x, a_start_y), (a_end_x, a_start_y), (0, 0, 255), 2)

                # represent line
                line2 = np.array([a_start_x, a_start_y, a_start_x, a_start_y])

            # fill the polygon created by line intersection
            # work with gray image
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            centroids = []
            polygon_points = []
            for i in range(len(contours)):
                # it aim to simplify the polyline by reducing the number of vertices based on epsilon value
                approx = cv2.approxPolyDP(contours[i], 0.015 * cv2.arcLength(contours[i], True), True)
                # only polygon with 4 vertex admit
                if len(approx) != 4:
                    flag = False

                polygon_points.append(approx)
                # find center of mass of polygon
                m = cv2.moments(approx)
                # m00 area of the polygon
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    centroids.append([cx, cy])
                else:
                    pass

        polygon_points = np.array(polygon_points)
        #polygon_points = self.align_polygon_edges(polygon_points)
        # number of polygon created, centroid of each polygon, polygon points
        return len(contours), np.array(centroids), polygon_points


    def set_field_permute(self, centroids, polygon_points, piece_color, canvas_height = 600, canvas_width = 600, number_of_pieces = 4):
        # print(f'centrois {centroids} \n polygon_points {polygon_points} \n piece color {piece_color} \n difference {difference}')
        # to randomized the pieces
        index = torch.randperm(4)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        # 4,2
        self.centroids = centroids[index,:]
        # 4,4,1,2
        self.polygon_points = polygon_points[index,:,:,:]
        #list 4
        self.piece_color = [piece_color[i] for i in index]
        # From the centroid we can have the all the other points of the polygon
        # 4,4,2
        self.difference = self.find_difference()

        self.index_shared_vertex = self.shared_vertex()
        self.number_of_pieces = number_of_pieces
        #self.centroids_distance = self.centroids_pairwise_distance()
        return index





