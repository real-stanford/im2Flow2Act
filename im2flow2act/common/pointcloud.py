from typing import Optional, Union

import cv2
import numpy as np


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)
    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3, :3], xyz_pts.T)  # apply rotation
    xyz_pts = xyz_pts + np.tile(
        rigid_transform[:3, 3].reshape(3, 1), (1, xyz_pts.shape[1])
    )  # apply translation
    return xyz_pts.T


def get_pointcloud(
    depth_img, color_img, segmentation_img, cam_intr, cam_pose=None, points=None
):
    """Get 3D pointcloud from depth image.

    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        color_img: HxWx3 uint8 array of color image
        segmentation_img: HxW int array of segmentation image
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix

    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
        color_pts: Nx1 int array of color points
    """

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x, pixel_y = np.meshgrid(
        np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h)
    )
    cam_pts_x = np.multiply(pixel_x - cam_intr[0, 2], depth_img / cam_intr[0, 0])
    cam_pts_y = np.multiply(pixel_y - cam_intr[1, 2], depth_img / cam_intr[1, 1])
    cam_pts_z = depth_img
    if points is not None:
        cam_pts = np.array([cam_pts_x, cam_pts_y, cam_pts_z]).transpose(1, 2, 0)
        cam_pts = cam_pts[points[:, 1], points[:, 0]].reshape(-1, 3)
    else:
        cam_pts = (
            np.array([cam_pts_x, cam_pts_y, cam_pts_z])
            .transpose(1, 2, 0)
            .reshape(-1, 3)
        )

    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)

    if color_img is None:
        color_pts = None
    else:
        if points is not None:
            color_pts = color_img
            color_pts = color_pts[points[:, 1], points[:, 0]].reshape(-1, 3)
        else:
            color_pts = color_img.reshape(-1, 3)

    segmentation_pts = (
        None if segmentation_img is None else segmentation_img.reshape(-1)
    )

    return cam_pts, color_pts, segmentation_pts


def meshwrite(filename, verts, colors, faces=None):
    """Save 3D mesh to a polygon .ply file.

    Args:
        filename: string; path to mesh file. (suffix should be .ply)
        verts: [N, 3]. Coordinates of each vertex
        colors: [N, 3]. RGB or each vertex. (type: uint8)
        faces: (optional) [M, 4]
    """
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    if faces is not None:
        ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write(
                "4 %d %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2], faces[i, 3])
            )

    ply_file.close()


def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]  # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    # Dot is matrix multiplication for array
    H = np.dot(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:  # Special reflection case
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t


def get_mujoco_pose_matrix(env, camera_id):
    rot_mat = env.mj_physics.data.cam_xmat[camera_id].reshape(3, 3)
    pos = env.mj_physics.data.cam_xpos[camera_id]
    forward = np.matmul(rot_mat, np.array([0, 0, -1]))
    forward = forward / np.linalg.norm(forward)

    up = np.matmul(rot_mat, np.array([0, 1, 0]))
    up = up / np.linalg.norm(up)

    u = up.copy()
    s = np.cross(forward, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, forward)
    view_matrix = np.array(
        [
            s[0],
            u[0],
            -forward[0],
            0,
            s[1],
            u[1],
            -forward[1],
            0,
            s[2],
            u[2],
            -forward[2],
            0,
            -np.dot(s, pos),
            -np.dot(u, pos),
            np.dot(forward, pos),
            1,
        ]
    )
    view_matrix = view_matrix.reshape(4, 4).T
    pose_matrix = np.linalg.inv(view_matrix)
    pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]
    return pose_matrix


def get_mujoco_intrinsic_matrix(env, height, width, camera_id):
    from dm_control.mujoco import Camera

    camera = Camera(env.mj_physics, camera_id=camera_id, height=height, width=width)
    f_x = -camera.matrices().focal[
        0, 0
    ]  # Negating to correct for the sign in the provided focal matrix
    f_y = camera.matrices().focal[1, 1]
    c_x = camera.matrices().image[0, 2]
    c_y = camera.matrices().image[1, 2]

    # Constructing the intrinsic matrix K
    intrinsic = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return intrinsic


def add_depth_noise(
    depth: Union[list, np.ndarray],
    color: Optional[Union[list, np.ndarray]] = None,
    gaussian_shifts=1,
    base_noise=100,
    std_noise=10,
    missing_depth_darkness_thres: int = 15,
) -> Union[list, np.ndarray]:
    """
    Add noise, holes and smooth depth maps according to the noise characteristics of the Kinect Azure sensor.
    https://www.mdpi.com/1424-8220/21/2/413

    For further realism, consider to use the projection from depth to color image in the Azure Kinect SDK:
    https://docs.microsoft.com/de-de/azure/kinect-dk/use-image-transformation

    :param depth: Input depth image(s) in meters
    :param color: Optional color image(s) to add missing depth at close to black surfaces
    :param missing_depth_darkness_thres: uint8 gray value threshold at which depth becomes invalid, i.e. 0
    :return: Noisy depth image(s)
    """

    if isinstance(depth, list) or hasattr(depth, "shape") and len(depth.shape) > 2:
        if color is None:
            color = len(depth) * [None]
        assert len(color) == len(depth), "Enter same number of depth and color images"
        return [
            add_depth_noise(d, c, missing_depth_darkness_thres)
            for d, c in zip(depth, color)
        ]

    # smoothing at borders
    depth = add_gaussian_shifts(depth, gaussian_shifts)

    # 0.5mm base noise, 1mm std noise @ 1m, 3.6mm std noise @ 3m
    depth += (
        base_noise / 10000 + np.maximum((depth - 0.5) * std_noise / 1000, 0)
    ) * np.random.normal(size=depth.shape)

    # Creates the shape of the kernel
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (3, 3))

    # Applies the minimum filter with kernel NxN
    min_depth = cv2.erode(depth, kernel)
    max_depth = cv2.dilate(depth, kernel)

    # missing depth at 0.8m min/max difference
    depth[abs(min_depth - max_depth) > 0.8] = 0

    # create missing depth at dark surfaces
    if color is not None:
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        depth[gray < missing_depth_darkness_thres] = 0

    return depth


def add_gaussian_shifts(
    image: Union[list, np.ndarray], std: float = 0.5
) -> Union[list, np.ndarray]:
    """
    Randomly shifts the pixels of the input depth image in x and y direction.

    :param image: Input depth image(s)
    :param std: Standard deviation of pixel shifts, defaults to 0.5
    :return: Augmented images
    """

    if isinstance(image, list) or hasattr(image, "shape") and len(image.shape) > 2:
        return [add_gaussian_shifts(img, std=std) for img in image]

    rows, cols = image.shape
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates
    xx = np.linspace(0, cols - 1, cols)
    yy = np.linspace(0, rows - 1, rows)

    # get xpixels and ypixels
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(image, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
