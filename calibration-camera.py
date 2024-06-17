import numpy as np
import cv2
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (9, 6)
# Define the real-world size of the squares in meters (e.g., 20mm = 0.02 meters)
SQUARE_SIZE = 0.024

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create arrays to store object points and image points from all the images
objpoints = []
imgpoints = []

# Prepare the object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE  # Scale the object points by the real size of the squares

# Get the paths of all the images
images = glob.glob('chessboard_calibration/*.png')  # Update with the path to your images

for image_file in images:
    print(image_file)
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points and image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    else:
        print(f"Checkerboard not detected in image: {image_file}")

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration results
np.savez(
    "CalibrationMatrix_college_cpt",
    Camera_matrix=mtx,
    distCoeff=dist,
    RotationalV=rvecs,
    TranslationV=tvecs
)

# Extract the focal lengths
fx = mtx[0, 0]
fy = mtx[1, 1]

print(f"Focal length in x direction (fx): {fx}")
print(f"Focal length in y direction (fy): {fy}")
