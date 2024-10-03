import numpy as np
from sklearn.cluster import DBSCAN


def get_buffer_size(root):
    keys = list(root.group_keys())
    return len(keys)


def max_distance_moved(points, t_threshold=-1):
    if t_threshold != -1:
        points = points[:, :t_threshold]

    N, T, _ = points.shape
    max_distances = np.zeros(N)

    for i in range(N):
        visible_points = points[i, points[i, :, 2] == 1, :2]
        if visible_points.shape[0] > 1:
            dist_matrix = np.linalg.norm(
                visible_points[:, np.newaxis] - visible_points, axis=2
            )
            max_dist = np.max(dist_matrix)
        else:
            max_dist = 0

        max_distances[i] = max_dist

    return max_distances


def create_uniform_grid(h, w, h_points=25, w_points=25):
    x = np.linspace(0, h - 1, h_points)
    y = np.linspace(0, w - 1, w_points)

    # Creating a meshgrid
    xx, yy = np.meshgrid(x, y)
    coordinates = np.dstack([xx, yy])
    sample_grid = coordinates.reshape(-1, 2)
    return sample_grid


def create_uniform_grid_from_bbox(bounding_box, grid_size):
    min_x, min_y, max_x, max_y = bounding_box
    grid_x, grid_y = grid_size

    # Create arrays of x and y coordinates
    x_coords = np.linspace(min_x, max_x, grid_x)
    y_coords = np.linspace(min_y, max_y, grid_y)

    # Create a meshgrid, which provides grid coordinates from the x and y arrays
    grid = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([grid[0].ravel(), grid[1].ravel()])

    return grid_points


def calculate_min_bbox(points, epsilon=0.5, min_samples=5, padding=0):
    # Using DBSCAN to cluster points
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    # Filtering points belonging to the main cluster (ignoring outliers)
    # Assume the largest cluster is the main cluster
    main_cluster = np.bincount(labels[labels >= 0]).argmax()
    filtered_points = points[labels == main_cluster]

    # Calculate the bounding box of the remaining points
    min_x, min_y = np.min(filtered_points, axis=0)
    max_x, max_y = np.max(filtered_points, axis=0)

    return (
        [min_x - padding, min_y - padding, max_x + padding, max_y + padding],
        filtered_points,
        labels,
    )


def filter_visiable_flow(flow, flow_format="TN3"):
    if flow_format == "TN3":
        visibility_first_step = flow[0, :, 2]
        # Identify points that are visible at the first time step
        visible_indices = np.where(visibility_first_step == 1)[0]
        # Filter out coordinates of these visible points
        visible_flow = flow[:, visible_indices, :]
    elif flow_format == "NT3":
        visibility_first_step = flow[:, 0, 2]
        # Identify points that are visible at the first time step
        visible_indices = np.where(visibility_first_step == 1)[0]
        # Filter out coordinates of these visible points
        visible_flow = flow[visible_indices, :, :]
    elif flow_format == "3NT":
        visibility_first_step = flow[2, :, 0]
        # Identify points that are visible at the first time step
        visible_indices = np.where(visibility_first_step == 1)[0]
        # Filter out coordinates of these visible points
        visible_flow = flow[:, visible_indices, :]
    else:
        raise NotImplementedError

    return visible_flow


def sort_flow_in_raster(flow, return_indices=False):
    # flow: (TxNx3)
    # Sorting the points based on x, y at time 0 (with x first, then y)
    sorted_indices = np.lexsort(
        (flow[0, :, 0], flow[0, :, 1])
    )  # sorting by x then y at time 0
    sorted_flow = flow[:, sorted_indices, :]
    if return_indices:
        return sorted_flow, sorted_indices
    return sorted_flow
