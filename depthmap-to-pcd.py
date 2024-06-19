import numpy as np
import cv2
import open3d as o3d


def depth_map_to_colored_point_cloud(depth_map, rgb_image):
    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Initialize an empty list to store points and colors
    points = []
    colors = []

    # Get the height and width of the depth map
    height, width = depth_map.shape

    # Iterate through each pixel in the depth map
    for i in range(height):
        for j in range(width):
            z = depth_map[i, j] * (-6)
            if z != 0:  # Ignore zero depth values
                x = j
                y = i
                points.append([x, y, z])

                # Get the color from the RGB image
                color = rgb_image[i, j] / 255.0  # Normalize to [0, 1]
                colors.append(color)

    # Convert points and colors to numpy arrays
    points = np.array(points) / 58.4
    colors = np.array(colors)

    # Set the points and colors in the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def save_point_cloud(point_cloud, filename):
    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud(filename, point_cloud)


# Example usage
depth_map_image_path = 'my_test/depthmap/IMG_7376_D.png'  # Path to the depth map image
rgb_image_path = 'my_test/depthmap/IMG_7376.png'  # Path to the RGB image

# Load the depth map as a grayscale image
depth_map = cv2.imread(depth_map_image_path, cv2.IMREAD_GRAYSCALE)

# Load the RGB image
rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)

# Ensure the RGB image and depth map have the same dimensions
assert depth_map.shape == rgb_image.shape[:2], "Depth map and RGB image must have the same dimensions"

point_cloud = depth_map_to_colored_point_cloud(depth_map, rgb_image)
point_cloud = point_cloud.voxel_down_sample(0.5)
save_point_cloud(point_cloud, 'my_test/depthmap/IMG_7376_D.ply')
