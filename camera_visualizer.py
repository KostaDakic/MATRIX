import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.spatial.transform import Rotation

def visualize_camera(camera_matrix, rvec, tvec, map_width, map_height):
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Camera position in world coordinates
    C = -np.matrix(R).T * np.matrix(tvec)

    # Create a Rotation object from the rotation vector

    R_camera_to_world, _ = cv2.Rodrigues(rvec)
    R_world_to_camera = R_camera_to_world.T
    z_axis, _ = cv2.Rodrigues(R_world_to_camera)
    # z_axis = R[:, 2]

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera position
    ax.scatter(C[0], C[1], C[2], c='r', s=100, label='Camera')

    # Plot camera orientation
    length = 5  # Length of the orientation vector
    ax.quiver(C[0], C[1], C[2], z_axis[0], z_axis[1], z_axis[2], length=length, color='g', label='Camera Direction')

    # Plot ground plane
    x = np.linspace(0, map_width, 10)
    y = np.linspace(0, map_height, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Position and Orientation')

    # Set equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), C[2].max()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = C[2].max() * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, mid_z + max_range)

    # Add legend
    ax.legend()

    # Print camera information
    print(f"Camera position: {C.flatten()}")
    print(f"Camera direction: {z_axis}")

    # Enable interactive rotation
    plt.ion()

    def on_move(event):
        if event.inaxes == ax:
            ax.view_init(elev=ax.elev, azim=ax.azim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()

    # Keep the plot open
    # input("Press Enter to close the plot...")

    print('Camera Position and Orientation')


