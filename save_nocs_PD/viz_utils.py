import numpy as np
import matplotlib.pyplot as plt
import cv2


def draw_bboxes_mpl_glow(img, img_pts, axes, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    color_ground = color
    n_lines = 8
    diff_linewidth = 1.05
    alpha_value = 0.03
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    for i, j in zip(range(4), range(4, 8)):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
            
    # draw axes
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 4) ## y last

    return img