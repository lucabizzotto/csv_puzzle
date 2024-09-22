"""
it contain utility function used in our project
"""
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


class Utility:
    @staticmethod
    def normalize_coordinate(center_coordinates, height, width, device):
        """
        transform coordinate into range -1, 1 center(0,0) top right corner(1,1) bottomleft(-1,-1)
        : param p1: center coordinate to normalize
        : param p2: height of the image
        : param p3: width of the image
        """
        flag = False
        if len(center_coordinates.shape) != 2:
            flag = True
            # save the shape
            original_shape = center_coordinates.shape
            center_coordinates = center_coordinates.reshape((-1, 2))
        if isinstance(center_coordinates,np.ndarray):
            center_coordinates = torch.from_numpy(center_coordinates)
        # compute the center of the image
        #                         x       y
        center = torch.tensor([width / 2, height / 2]).to(device)
        center_coordinates = center_coordinates.to(device)
        # move the center to 0,0
        center_coordinates = torch.stack([center_coordinates[:, 0] - center[0], - center_coordinates[:, 1] + center[1]],
                                         dim=1)

        # normalize
        center_coordinates = center_coordinates / center
        if flag:
            center_coordinates = center_coordinates.reshape(original_shape)
        return center_coordinates

    @staticmethod
    def denormalize_coordinate(normalize_coordinate, height, width, device):
        """
        transform coordinate back to original scale from range -1, +1
        : param p1: coordinate to denormalize
        : param p2: height of the image
        : param p3: width of the image
        """
        flag = False
        if len(normalize_coordinate.shape) != 2:
            flag = True
            # save the shape
            original_shape = normalize_coordinate.shape
            normalize_coordinate = normalize_coordinate.reshape((-1, 2))

        normalize_coordinate = normalize_coordinate.to(device)
        center = torch.tensor([width / 2, height / 2]).to(device)
        # denormalize
        denormalize = torch.stack([(normalize_coordinate[:, 0] * center[0]) + center[0],
                                   -1 * (normalize_coordinate[:, 1] * center[1]) + center[1]], dim=1)
        if flag:
            denormalize = denormalize.reshape(original_shape)
        return denormalize

    @staticmethod
    def plot_scatter_point(points, puzzle, device):
        """
        comparison plots
        """

        # need to transfer to cpu to plot data
        points = points.cpu()
        ground_thruth = puzzle.get_centroids()

        # normalize -1 +1
        ground_thruth = Utility.normalize_coordinate(ground_thruth, puzzle.get_height(), puzzle.get_width(), device)
        colors = puzzle.get_piece_color()

        # plot
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots(1,10, figsize=(20, 20))

        f.tight_layout(pad=2)
        ax = ax.flat
        for enum, point in enumerate(points):
            ax[enum].set_xlim(-1, 1)
            ax[enum].set_ylim(-1, 1)
            ax[enum].scatter(point[:, 0], point[:, 1], c=colors / 255, marker="^")
            ax[enum].scatter(ground_thruth[:, 0], ground_thruth[:, 1], c=colors / 255, marker="o")
            ax[enum].set_title(f"Iter: {1000 - (enum + 1) * 50}", fontsize=10)

        # set common label for subplot
        # handles, labels = ax[0].get_legend_handles_labels()
        f.legend(["Estimated", "Groundtruth"])
        plt.show()

    """
    new_polygon_points()
    With the updated centroid points for each puzzle piece and the puzzle itself, we calculate the displacement of 
    each centroid. Then, we reconstruct the position of each vertex for every puzzle piece 
    """

    @staticmethod
    def new_polygon_points(batch_noise, puzzle, device):
        """

        :param batch_noise: center coordinate (denormalize coordinate)
        :param puzzle:
        :param device:
        :return: the canvas of the puzzle to respect the new piece position
        """


        t = Utility.vertex_point(batch_noise, puzzle, device)
        # extreme to create canvas of right dimension
        a, b = int(torch.max(t[:, :, :, 0])), int(torch.max(t[:, :, :, 1]))
        # consider possible presence of negative value
        a_min, b_min = int(torch.min(t[:, :, :, 0])), int(torch.min(t[:, :, :, 1]))
        # case smaller size
        a, b = max(a, puzzle.get_width()), max(b, puzzle.get_height())
        dx = 0
        dy = 0
        if a_min < 0:
            a = a - a_min
            dx = -a_min
        if b_min < 0:
            b = b - b_min
            dy = - b_min
        # create canvas
        canvas = np.ones((b, a, 3), dtype=np.uint8) * 255
        colors = puzzle.get_piece_color()
        t = t.cpu()
        # fill the polygon
        for enum, small_t in enumerate(t):
            """
            canvas = cv2.fillPoly(canvas, [np.int32(small_t.detach().numpy()) + np.int32(np.array([dx, dy]))], color=(
                int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                int(puzzle.get_piece_color()[enum][2])))
            canvas = cv2.polylines(canvas, [np.int32(small_t.numpy()) + np.int32(np.array([dx, dy]))], isClosed=True,
                                   color=(
                                       int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                                       int(puzzle.get_piece_color()[enum][2])), thickness=4)

            """
            """
            canvas = cv2.polylines(canvas, [np.int32(small_t.numpy()) + np.int32(np.array([dx, dy]))], isClosed=True,
                                   color=(
                                       int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                                       int(puzzle.get_piece_color()[enum][2])), thickness=4)
            """

            canvas = cv2.fillPoly(canvas, [np.int32(small_t.detach().numpy()) + np.int32(np.array([dx, dy]))], color=(
                int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                int(puzzle.get_piece_color()[enum][2])))




        return canvas, dx, dy


    @staticmethod
    def new_polygon_points_line(batch_noise, puzzle, device):
        """

        :param batch_noise: center coordinate (denormalize coordinate)
        :param puzzle:
        :param device:
        :return: the canvas of the puzzle to respect the new piece position
        """


        t = Utility.vertex_point(batch_noise, puzzle, device)
        # extreme to create canvas of right dimension
        a, b = int(torch.max(t[:, :, :, 0])), int(torch.max(t[:, :, :, 1]))
        # consider possible presence of negative value
        a_min, b_min = int(torch.min(t[:, :, :, 0])), int(torch.min(t[:, :, :, 1]))
        # case smaller size
        a, b = max(a, puzzle.get_width()), max(b, puzzle.get_height())
        dx = 0
        dy = 0
        if a_min < 0:
            a = a - a_min
            dx = -a_min
        if b_min < 0:
            b = b - b_min
            dy = - b_min
        # create canvas
        canvas = np.ones((b, a, 3), dtype=np.uint8) * 255
        colors = puzzle.get_piece_color()
        t = t.cpu()
        # fill the polygon
        for enum, small_t in enumerate(t):
            """
            canvas = cv2.fillPoly(canvas, [np.int32(small_t.detach().numpy()) + np.int32(np.array([dx, dy]))], color=(
                int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                int(puzzle.get_piece_color()[enum][2])))
            canvas = cv2.polylines(canvas, [np.int32(small_t.numpy()) + np.int32(np.array([dx, dy]))], isClosed=True,
                                   color=(
                                       int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                                       int(puzzle.get_piece_color()[enum][2])), thickness=4)

            """

            canvas = cv2.polylines(canvas, [np.int32(small_t.numpy()) + np.int32(np.array([dx, dy]))], isClosed=True,
                                   color=(
                                       int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                                       int(puzzle.get_piece_color()[enum][2])), thickness=4)
            """

            canvas = cv2.fillPoly(canvas, [np.int32(small_t.detach().numpy()) + np.int32(np.array([dx, dy]))], color=(
                int(puzzle.get_piece_color()[enum][0]), int(puzzle.get_piece_color()[enum][1]),
                int(puzzle.get_piece_color()[enum][2])))
            """



        return canvas, dx, dy

    @ staticmethod
    def get_polygon_from_center(center, difference):
        """
        from center point get back the vertex componing each puzzle piece
        :param center:
        :param difference:
        :return:
        """
        center
        pp = torch.empty(4, 4, 2)
        for enum, d in enumerate(difference):
            p = d + center[enum]
            pp[enum] = p
        return pp

    """
    given a puzzle it create a mask that allow to divide background and foreground
    """

    @staticmethod
    def create_mask(canvas):
        yuv_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV)
        # Normalize UV channels between [-0.5 ... 0.5]
        UV = yuv_image[:, :, 1:3] / 255.0 - 0.5
        # we choosed 0.4 -0.4 based on the UV plane (green)
        # calculate distance
        mask = np.round(np.sqrt((UV[:, :, 0] - (-0.0)) ** 2 + (UV[:, :, 1] - (-0.0)) ** 2), 5)
        # constant to separate foreground-background
        mask[mask == 0.00277] = 255
        # foreground black
        mask[mask != 255] = 0
        return torch.from_numpy(mask)

    """
    given two images with two different size make the same
    """

    @staticmethod
    def resize(image1, image2):
        # find the smaller and bigger
        if image1.shape != image2.shape:
            max_height = max(image1.shape[0], image2.shape[0])
            max_width = max(image1.shape[1], image2.shape[1])
            # how much do we have to reshape them
            image1_padding = (max_height - image1.shape[0]) // 2, (max_width - image1.shape[1]) // 2
            image2_padding = (max_height - image2.shape[0]) // 2, (max_width - image2.shape[1]) // 2
            # padding
            resize_img1 = cv2.copyMakeBorder(image1, image1_padding[0], image1_padding[0], image1_padding[1],
                                             image1_padding[1],
                                             cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])
            resize_img2 = cv2.copyMakeBorder(image2, image2_padding[0], image2_padding[0], image2_padding[1],
                                             image2_padding[1],
                                             cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])
            # case still not equal for odd number division by 2
            if resize_img1.shape != resize_img2.shape:
                max_height = max(resize_img1.shape[0], resize_img2.shape[0])
                max_width = max(resize_img1.shape[1], resize_img2.shape[1])
                image1_padding = (max_height - resize_img1.shape[0]), (max_width - resize_img1.shape[1])
                image2_padding = (max_height - resize_img2.shape[0]), (max_width - resize_img2.shape[1])
                resize_img1 = cv2.copyMakeBorder(resize_img1, image1_padding[0], 0, image1_padding[1],
                                                 0,
                                                 cv2.BORDER_CONSTANT,
                                                 value=[255, 255, 255])
                resize_img2 = cv2.copyMakeBorder(resize_img2, image2_padding[0], 0, image2_padding[1],
                                                 0,
                                                 cv2.BORDER_CONSTANT,
                                                 value=[255, 255, 255])

            return resize_img1, resize_img2

        return image1, image2

    @staticmethod
    def resize_batch(image, resize_shape):

        # find the smaller and bigger
        if image.shape[0:2] != resize_shape:
            max_height = resize_shape[0]
            max_width = resize_shape[1]
            # how much do we have to reshape them
            image_padding = (max_height - image.shape[0]) // 2, (max_width - image.shape[1]) // 2
            # padding
            resize_img = cv2.copyMakeBorder(image, image_padding[0], image_padding[0], image_padding[1],
                                            image_padding[1],
                                            cv2.BORDER_CONSTANT,
                                            value=[255, 255, 255])

            # case still not equal for odd number division by 2
            if resize_img.shape[0:2] != resize_shape:
                max_height = resize_shape[0]
                max_width = resize_shape[1]
                image_padding = (max_height - resize_img.shape[0]), (max_width - resize_img.shape[1])
                resize_img = cv2.copyMakeBorder(resize_img, image_padding[0], 0, image_padding[1],
                                                0,
                                                cv2.BORDER_CONSTANT,
                                                value=[255, 255, 255])
            return resize_img
        return image


    @staticmethod
    def vertex_point(centers, puzzle, device):
        """

        :param centers: center of each puzzle piece (denormalize coordinate)
        :param puzzle:
        :param device:
        :return: return the vertex points of each piece
        """

        new_centroids = centers.to(device)
        # displacement from original centroids
        displacement = new_centroids - puzzle.get_centroids().to(device)
        #print(puzzle.get_centroids(), "puzzle.get_centroids()\n", puzzle.get_difference(), "puzzle.get_difference()\n")
        # polygon points
        #PP = puzzle.get_difference().to(device) + puzzle.get_centroids().to(device)
        center = puzzle.get_centroids().to(device)
        differences = puzzle.get_difference().to(device)
        #PP = torch.empty(4, 4, 2).to(device)
        # number of pieces composing rh puzzle, number of vertex for each polygon piece, cordinate xy
        PP = torch.empty(puzzle.get_number_puzzle_pieces(), 4, 2).to(device)
        for enum, i in enumerate(differences):
            v = i + center[enum]
            PP[enum] = v

        t = torch.tensor([]).to(device)
        for pp, n in zip(PP, displacement):
            # new point polygon
            r = pp + n
            t = torch.cat((t, r))
        # reshape correctly
        t = t.reshape(4, 4, 1, 2)
        # vertex point of each puzzle piece after displacements
        return t

    @staticmethod
    def normalize_shared_point_gt_est(center_estimated, puzzle, device):
        """
        :param center_estimated: center position estimated (normalized)
        :param puzzle: puzzle
        :param device: device
        :return: gt represent normalize cordinate of the groudn truth of shered vertex,  e represent normalize cordinate of the estimated of shared vertex
        """

        # normalize coordinate
        # polygon vertex of the ground thruth
        vertices_origin = Utility.vertex_point(puzzle.get_centroids(), puzzle,
                                                       device)
        # normalize it
        vertices_origin = Utility.normalize_coordinate(vertices_origin, puzzle.get_height(), puzzle.get_width(),
                                                               device)
        # denormalize
        center_estimated_denormalize = Utility.denormalize_coordinate(center_estimated, puzzle.get_height(),
                                                                              puzzle.get_width(), device)
        # compute the vertices of each puzzle piece
        vertices = Utility.vertex_point(center_estimated_denormalize, puzzle, device)
        # normalize it
        vertices_normalized = Utility.normalize_coordinate(vertices, puzzle.get_height(), puzzle.get_width(),
                                                                   device)

        t = []
        e = []
        # get index of vertex suppose to be shared
        # shared vertex is a list of index each element contain the index of vertex suppose to be shared
        shared_vertex_index = puzzle.get_index_shared_vertex()
        for i in shared_vertex_index:
            for j in i:
                # the estimated one
                new = vertices_normalized[j[0], j[1], j[2]]
                # ground thruth
                old = vertices_origin[j[0], j[1], j[2]]
                t.append(old)
                e.append(new)
        # transform to tensor
        t = torch.cat(t, dim=0).reshape((-1, 2))
        e = torch.cat(e, dim=0).reshape((-1, 2))
        # ground thruth, estimated
        return t, e

    @staticmethod
    def random_start(puzzle, device):
        """

        :param device:
        :return: random starting position constrained in puzzle boundary
        """
        start = []
        for i in range(4):
            # get random position
            cor = torch.cat(
                (torch.randint(0, puzzle.canvas_width, (1, 1)), torch.randint(0, puzzle.canvas_height, (1, 1))),
                dim=1)
            # normalize cordinate
            start.append(Utility.normalize_coordinate(cor, puzzle.get_height(), puzzle.get_width(), device))
        start = torch.cat(start)
        return start

    @staticmethod
    def plot_shared_vertex(center_estimated_denormalize, puzzle, device, v1 = None, v2 = None):
        """
        help to visualize the shared vertex
        :param center_estimated_denormalize:
        :param puzzle:
        :param device:
        :return:
        """
        sns.set_theme(style="whitegrid")
        # ground thruth
        origin = Utility.normalize_coordinate(puzzle.get_centroids(), puzzle.get_height(), puzzle.get_width(),
                                                      device)
        # normal cordinate
        vertices_origin = Utility.vertex_point(puzzle.get_centroids(), puzzle,
                                                       device)
        # normalized cordinate
        vertices_origin = Utility.normalize_coordinate(vertices_origin, puzzle.get_height(), puzzle.get_width(),
                                                               device)

        # compute the vertices of each puzzle piece
        vertices = Utility.vertex_point(center_estimated_denormalize, puzzle, device)
        # normalize it
        vertices_normalized = Utility.normalize_coordinate(vertices, puzzle.get_height(), puzzle.get_width(),
                                                                   device)

        img, dx, dy  = Utility.new_polygon_points(center_estimated_denormalize, puzzle, device)
        shared_vertex_index = puzzle.get_index_shared_vertex()

        fig, ax = plt.subplots(1, 4)
        if v1 != None:
            fig.suptitle( "mse :" + str(torch.nn.functional.mse_loss(v1, v2).item()), fontsize=16)
        color = ["green", "red", "purple", "black", "gold"]

        for enum, i in enumerate(shared_vertex_index):
            for j in i:
                # the estimated one
                new = vertices_normalized[j[0], j[1], j[2]]
                ax[0].scatter(new[0], new[1], color=color[enum])

        # display normalize points ground thruth and estimate
        for enum, e in enumerate(vertices_normalized):
            # estimated
            ax[0].scatter(e[:, :, 0], e[:, :, 1], color=color[enum], marker="+")

        ax[0].set_xlim(-2.8, 2.8)
        ax[0].set_ylim(-2.8, 2.8)
        ax[1].imshow(img)
        ax[2].imshow(puzzle.get_puzzle())
        #########
        if v1 != None:
            color = np.random.rand(20, 3)
            for enum, (q,w) in enumerate(zip(v1,v2)):
                ax[3].scatter(q[0], q[1], color= color[enum], marker="1")
                ax[3].scatter(w[0], w[1], color= color[enum], marker="*")

            ax[3].set_xlim(-2.8, 2.8)
            ax[3].set_ylim(-2.8, 2.8)

        #########
        plt.show()
        plt.pause(3)


    @staticmethod
    def relative_shared_vertex(center_estimated, puzzle, device):
        """
        given the center estimate coordinate return the list of tensor we want to minimize the distance
        :param center_estimated: in normalized coordinate
        :param puzzle:
        :param device:
        :return: the list of tensor we want to minimize the distance l1[i] should be same as l2[i] in normalize coordinate
        """
        # denormalize
        center_estimated_denormalize = Utility.denormalize_coordinate(center_estimated, puzzle.get_height(),
                                                                              puzzle.get_width(), device)
        # compute the vertices of each puzzle piece
        vertices = Utility.vertex_point(center_estimated_denormalize, puzzle, device)
        # normalize it
        vertices_normalized = Utility.normalize_coordinate(vertices, puzzle.get_height(), puzzle.get_width(),
                                                                   device)
        # get index of shared vertex
        shared_vertex_index = puzzle.get_index_shared_vertex()

        overall = []
        for enum, i in enumerate(shared_vertex_index):
            # all vertex here should be at same position
            same = []
            for j in i:
                same.append(vertices_normalized[j[0], j[1], j[2]])
            # create the combination of point to compute distance
            overall.append([(same[a], same[b]) for a in range(len(same)) for b in range(a, len(same)) if a != b])
        l1 = []
        l2 = []
        # list of list
        # overall: list[[(point1,point2), (point1, point3), (point2, point3)], ...., [(),...,()]]
        for enum, og in enumerate(overall):
            # tupla
            for s in og:
                l1.append(s[0])
                l2.append(s[1])
        l1 = torch.cat(l1, dim=0)
        l2 = torch.cat(l2, dim=0)
        l1 = l1.reshape((-1, 2))
        l2 = l2.reshape((-1, 2))
        return l1, l2

    @staticmethod
    def IoU(box1, box2):
        """
        Compute Intersection over Union (IoU) of two bounding boxes.
        :param box1: Tuple or list containing (x1, y1, x2, y2) of the first bounding box
        :param box2: Tuple or list containing (x1, y1, x2, y2) of the second bounding box
        :return: Intersection over Union (IoU) value
        """
        # Determine coordinates of intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate area of intersection rectangle
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate areas of individual bounding boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Compute IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)

        return iou

    @staticmethod
    def smooth_min(a, b, eps=1e-6):
        return torch.log(torch.exp(a) + torch.exp(b) - eps)

    @staticmethod
    def smooth_max(a, b, eps=1e-6):
        return -torch.log(torch.exp(-a) + torch.exp(-b) - eps)

    @staticmethod
    def get_bl_tr(vertices):
        """
        return vertices for bounding box needed to calculate Iou between two boxes
        :param vertices: pass 4 vertex of the rectangol
        :return:
        """
        max_x =  torch.logsumexp(vertices[:,0], dim = 0) #torch.log(torch.exp(vertices[:,0]).sum(dim=0) + 1e-6)
        max_y = torch.logsumexp(vertices[:,1], dim = 0)
        min_x =  -torch.logsumexp(-vertices[:,0], dim = 0)
        min_y =  -torch.logsumexp(-vertices[:,1], dim = 0)
        bl = torch.tensor([min_x, min_y])

        tr = torch.tensor([max_x, max_y])

        indices_bl = torch.where(torch.all(vertices == bl, dim=1))
        indices_tr = torch.where(torch.all(vertices == tr, dim=1))

        # [min_x, min_y, max_x, max_y]
        # re = torch.tensor([ vertices[indices_bl,0], vertices[indices_bl,1], vertices[indices_tr,0], vertices[indices_tr,1] ])
        re = torch.tensor( [min_x, min_y, max_x, max_y], requires_grad=True)
        #vertices[0, 0], vertices[0, 1], vertices[2, 0], vertices[2, 1]
        return re

    @staticmethod
    def overall_Iou_puzzle(vertices, img=None):
        """
        :param vertices: [N, 2]
        :return: sum overlap of all pair of puzzle pieces
        """
        overall = 0.
        for i in range(len(vertices)):
            for j in range(i, len(vertices)):
                if i != j:
                    box_1 = Utility.get_bl_tr(vertices[i, :, :, :].reshape(4, 2))
                    box_2 = Utility.get_bl_tr(vertices[j, :, :, :].reshape(4, 2))
                    overall += Utility.IoU(box_1, box_2).item()
                    if img is not None:
                        plt.imshow(img)
                        plt.scatter(vertices[i, :, :, 0], vertices[i, :, :, 1])
                        plt.scatter(vertices[j, :, :, 0], vertices[j, :, :, 1])
                        plt.title(f'IoU {Utility.IoU(box_1, box_2).item()}')
                        plt.show()
                        plt.pause(2)
        return overall

    @staticmethod
    def centroids_pairwise_distance(puzzle, device):
        """
        :return: the pariwise disatcne betwen centroids of each puzzle piece
        """
        centroids = puzzle.get_centroids()
        centroids_normalize = Utility.normalize_coordinate(centroids, puzzle.get_height(), puzzle.get_width(), device)
        distance = []
        for i in range(centroids.shape[0]):
            for j in range(i + 1, centroids.shape[0]):
                a = centroids_normalize[i].float()
                b = centroids_normalize[j].float()
                norm = torch.linalg.vector_norm(a - b  , 2)
                distance.append(norm)
        distance = torch.stack(distance)
        return distance




    @staticmethod
    # delete later
    def relative_shared_vertex_1(center_estimated, puzzle, device):
        """
        given the center estimate coordinate return the list of tensor we want to minimize the distance
        :param center_estimated: in normalized coordinate
        :param puzzle:
        :param device:
        :return: the list of tensor we want to minimize the distance l1[i] should be same as l2[i]
        """
        # denormalize
        center_estimated_denormalize = Utility.denormalize_coordinate(center_estimated, puzzle.get_height(),
                                                                      puzzle.get_width(), device)
        # compute the vertices of each puzzle piece
        vertices = Utility.vertex_point(center_estimated_denormalize, puzzle, device)
        # normalize it
        vertices_normalized = Utility.normalize_coordinate(vertices, puzzle.get_height(), puzzle.get_width(),
                                                           device)
        # get index of shared vertex
        shared_vertex_index = puzzle.get_index_shared_vertex()

        overall = []
        overall_den = []
        for enum, i in enumerate(shared_vertex_index):
            # all vertex here should be at same position
            same = []
            same_den = []
            for j in i:
                same.append(vertices_normalized[j[0], j[1], j[2]])
                same_den.append(vertices[j[0], j[1], j[2]])
            # create the combination of point to compute distance
            overall.append([(same[a], same[b]) for a in range(len(same)) for b in range(a, len(same)) if a != b])
            overall_den.append([(same_den[a], same_den[b]) for a in range(len(same)) for b in range(a, len(same)) if a != b])
        l1 = []
        l2 = []

        # list of list
        # overall: list[[(point1,point2), (point1, point3), (point2, point3)], ...., [(),...,()]]
        for enum, og in enumerate(overall):
            # tupla
            for s in og:
                l1.append(s[0])
                l2.append(s[1])
        l1 = torch.cat(l1, dim=0)
        l2 = torch.cat(l2, dim=0)
        l1 = l1.reshape((-1, 2))
        l2 = l2.reshape((-1, 2))

        l1_den = []
        l2_den = []
        for enum, og in enumerate(overall_den):
            # tupla
            for s in og:
                l1_den.append(s[0])
                l2_den.append(s[1])
        l1_den = torch.cat(l1_den, dim=0)
        l2_den = torch.cat(l2_den, dim=0)
        l1_den = l1_den.reshape((-1, 2))
        l2_den = l2_den.reshape((-1, 2))

        return l1, l2, l1_den,l2_den