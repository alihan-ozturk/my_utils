import numpy as np
import cv2
import math


def grid_lines(src, clahe_transform=None, median_blur_ksize=11, canny_th1=30, canny_th2=60, hough_lines_th=500):
    h, w = src.shape
    v_lines = []
    h_lines = []
    src = cv2.medianBlur(src, median_blur_ksize)
    if clahe_transform is not None:
        src = clahe_transform.apply(src)
    src = cv2.Canny(src, canny_th1, canny_th2)
    lines = cv2.HoughLines(src, 1, np.pi / 180, hough_lines_th, None, 0, 0)
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)

        dist = abs(theta - np.array([0, 1.57]))
        arg = np.argmin(dist)

        if arg:
            h_lines.append(y0)  # ((0, y0), (w, y0))
        else:
            v_lines.append(x0)  # ((x0, 0), (x0, h))

    return np.array(v_lines), np.array(h_lines)


def find_closest_lines(x, y, v_x, h_y, xmax, ymax, trim):
    v_x_xmin = v_x[v_x <= x - trim]
    v_x_xmax = v_x[v_x > x + trim]
    h_y_ymin = h_y[h_y <= y - trim]
    h_y_ymax = h_y[h_y > y + trim]
    if len(v_x_xmin) == 0:
        v_x_xmin_value = 0
    else:
        v_x_xmin_value = v_x_xmin.max()

    if len(v_x_xmax) == 0:
        v_x_xmax_value = xmax
    else:
        v_x_xmax_value = v_x_xmax.min()

    if len(h_y_ymin) == 0:
        h_y_ymin_value = 0
    else:
        h_y_ymin_value = h_y_ymin.max()

    if len(h_y_ymax) == 0:
        h_y_ymax_value = ymax
    else:
        h_y_ymax_value = h_y_ymax.min()

    return v_x_xmin_value, v_x_xmax_value, h_y_ymin_value, h_y_ymax_value


def rearrange(array, th):
    print(th)
    new_array = []
    skip = np.diff(array) <= th
    if skip.sum() == 0:
        return array.astype(int)
    for i in range(len(array) - 1):
        if skip[i]:
            new_array.append((array[i] + array[i + 1]) / 2)
        elif i > 0 and skip[i - 1]:
            continue
        else:
            new_array.append(array[i])
    if not skip[-1]:
        new_array.append(array[-1])

    return rearrange(np.array(new_array), th)


def add_lines(array):
    diff = np.diff(array)
    step = np.median(diff)
    add = diff > step * 1.2
    new_array = [array[0]]
    for i in range(1, len(array) - 1):
        new_array.append(array[i])
        if add[i]:
            new_array.append(new_array[-1] + step)
    new_array.append(array[-1])
    return np.array(new_array, dtype=int)
