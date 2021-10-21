import cv2
import numpy as np

def process(frame):
    # 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    canny = cv2.Canny(blur, 50, 150)
    # 不规则ROI区域截取
    height, width = canny.shape
    points = np.array([[(0, height), (460, 325), (520, 325), (width, height)]], np.int32)
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, points, (255, 255, 255))
    img = cv2.bitwise_and(canny, mask)

    # 霍夫直线检测
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=20)

    # 车道计算
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k > 0:
                right_lines.append(line)
            else:
                left_lines.append(line)

    # 去除异常数据
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)

    # 最小二乘法拟合车道，并绘制车道
    if len(left_lines) > 0:
        left = least_squares_fit(left_lines, 325, height)
        cv2.line(frame, left[0], left[1], (0, 255, 0), 3)
    if len(right_lines) > 0:
        right = least_squares_fit(right_lines, 325, height)
        cv2.line(frame, right[0], right[1], (0, 255, 0), 3)

    return frame


def clean_lines(lines, threshold):
    if len(lines) == 0:
        return
    ks = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    mean = np.mean(ks)
    while len(lines) > 0:
        diff = [abs(k - mean) for k in ks]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            lines.pop(idx)
            ks.pop(idx)
        else:
            break


def least_squares_fit(lines, ymin, ymax):
    x = [x1 for line in lines for x1, y1, x2, y2 in line]
    y = [y1 for line in lines for x1, y1, x2, y2 in line]
    x += [x2 for line in lines for x1, y1, x2, y2 in line]
    y += [y2 for line in lines for x1, y1, x2, y2 in line]

    fit_fn = np.poly1d(np.polyfit(y, x, 1))
    return [(int(fit_fn(ymin)), ymin), (int(fit_fn(ymax)), ymax)]


if __name__ == '__main__':
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret or cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow("", process(frame))
