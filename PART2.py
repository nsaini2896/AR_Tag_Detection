import cv2
import numpy as np


# user input
print("Choose Tag Video")
print("0 for Tag0")
print("1 for Tag1")
print("2 for Tag2")
ent = int(input("Input is: "))
if ent == 0:
    vid = cv2.VideoCapture('Tag0.mp4')
elif ent == 1:
    vid = cv2.VideoCapture('Tag1.mp4')
elif ent == 2:
    vid = cv2.VideoCapture('Tag2.mp4')

else:
    print("Tag absent")
    exit(0)
image1 = cv2.imread('Testudo.png')


def tag_matrix(image):
    dimension_tag = image.shape  # Shape of the image
    ht_img = dimension_tag[0]
    wt_img = dimension_tag[1]
    bit_ht = int((ht_img / 8))
    bit_wt = int(wt_img / 8)
    a = 0
    b = 0
    ar_tag = np.empty((8, 8))  # start the 8X8 matrix
    for i in range(0, ht_img, bit_ht):
        b = 0
        for j in range(0, wt_img, bit_wt):
            black_cell = 0
            white_cell = 0
            for x in range(0, bit_ht - 1):
                for y in range(0, bit_wt - 1):
                    if (image[i + x][j + y] == 0):
                        black_cell = black_cell + 1
                    else:
                        white_cell = white_cell + 1

            if (
                    white_cell >= black_cell):
                ar_tag[a][b] = 1
            else:
                ar_tag[a][b] = 0
            b = b + 1
        a = a + 1
    return ar_tag


# Check for the tag orientation
def Tag_check(ar_tag):
    tag_generate = tag_matrix(ar_tag)
    if (tag_generate[2][2] == 0 and tag_generate[2][5] == 0 and tag_generate[5][2] == 0 and tag_generate[5][
        5] == 1):
        ang_rotate = 0
    elif (tag_generate[2][2] == 1 and tag_generate[2][5] == 0 and tag_generate[5][2] == 0 and tag_generate[5][
        5] == 0):
        ang_rotate = 180
    elif (tag_generate[2][2] == 0 and tag_generate[2][5] == 1 and tag_generate[5][2] == 0 and tag_generate[5][
        5] == 0):
        ang_rotate = 90
    elif (tag_generate[2][2] == 0 and tag_generate[2][5] == 0 and tag_generate[5][2] == 1 and tag_generate[5][
        5] == 0):
        ang_rotate = -90
    else:
        ang_rotate = None

    if (ang_rotate == None):
        flag = False
        return flag, ang_rotate
    else:
        flag = True
        return flag, ang_rotate


# Homography of the April Tag
def find_homography(frame1, frame2):
    A = []

    for i in range(0, len(frame2)):
        x, y = frame1[i][0], frame1[i][1]
        u, v = frame2[i][0], frame2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :] / Vh[-1, -1]
    h = np.reshape(l, (3, 3))
    return h


def orientation(list):
    count = 0
    top_left, top_right, bottom_left, bottom_right = 0, 0, 0, 0
    x = list[0][1]
    y = list[0][0]
    corner = ''
    # Condition to extract AR Tag corners and the index of the corners to be fed into 'corner_four' list
    if thresh[x - 10][y - 10] == 255:
        count = count + 1
        top_left = 1
    if thresh[x + 10][y + 10] == 255:
        count = count + 1
        bottom_right = 1
    if thresh[x - 10][y + 10] == 255:
        count = count + 1
        top_right = 1
    if thresh[x + 10][y - 10] == 255:
        count = count + 1
        bottom_left = 1
    if count == 3:
        if bottom_right == 0:
            corner = 'TL'
        elif bottom_left == 0:
            corner = 'TR'
        elif top_right == 0:
            corner = 'BL'
        elif top_left == 0:
            corner = 'BR'
    return y, x, corner


def reorient(angle, list):
    corner_actual = [[], [], [], []]
    if angle == 0:
        corner_actual = list
    elif angle == 90:
        corner_actual = [list[2], list[0], list[3], list[1]]
    elif angle == -90:
        corner_actual = [list[1], list[3], list[0], list[2]]
    elif angle == 180:
        corner_actual = [list[3], list[2], list[1], list[0]]
    return corner_actual


K = np.array(
    [[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T

if (vid.isOpened() == False):
    print("Error opening video file")

while (vid.isOpened()):
    ret, frame = vid.read()  # Capture frame-by-frame
    if ret == True:
        smooth = cv2.GaussianBlur(frame, (7, 7), 0)  # Smoothing # 5,5 kernel # border= 0
        gray = cv2.cvtColor(smooth,
                            cv2.COLOR_BGR2GRAY)  # cv2.cvtColor() method is used to convert an image from one color space to another.

        ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # find contours present in the video
        contours, hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        corners_list = []
        blank = []
        for i in hie[0]:
            four_contour_list = []
            if i[3] == 1:
                blank.append(i[3])

            for c in blank:
                if cv2.contourArea(contours[c]) > 750:

                    epsilon = 0.1 * cv2.arcLength(contours[c], True)
                    contour_corners = cv2.approxPolyDP(contours[c], epsilon, True)

                    for j in contour_corners:
                        y, x, corner = orientation(j)
                        four_contour_list.append([y, x, corner])

                if len(four_contour_list) == 4:
                    corners_list.append(four_contour_list)

        if corners_list != []:
            for i in range(0, len(corners_list)):
                list = [0, 0, 0, 0]  # to put x and y corner values in a list
                for value in corners_list[i]:
                    if value[-1] == 'TL':
                        list[0] = value[0:2]
                    elif value[-1] == 'TR':
                        list[1] = value[0:2]
                    elif value[-1] == 'BL':
                        list[2] = value[0:2]
                    elif value[-1] == 'BR':
                        list[3] = value[0:2]

                if 0 not in list:
                    H = find_homography(list, [[0, 0], [199, 0], [0, 199], [199, 199]])
                    H_inv = np.linalg.inv(H)
                    black = np.zeros((200, 200))
                    for a in range(0, 200):
                        for b in range(0, 200):
                            x, y, z = np.matmul(H_inv, [a, b, 1])
                            y_dash = int(y / z)
                            x_dash = int(x / z)
                            if (1080 > y_dash > 0) and (1920 > x_dash > 0):
                                black[a][b] = thresh[y_dash][x_dash]
                    flag, angle = Tag_check(black)

                    if flag:
                        actual_corners = reorient(angle, list)

                        image1 = cv2.resize(image1, (90, 90))
                        H_new = find_homography(actual_corners, [[0, 0], [0, 89], [89, 0], [89, 89]])
                        H_inv = np.linalg.inv(H_new)

                        for j in range(0, 90):
                            for k in range(0, 90):
                                x_Testudo, y_Testudo, z_Testudo = np.matmul(H_inv, [j, k, 1])
                                y_Testudo_dash = int(y_Testudo / z_Testudo)
                                x_Testudo_dash = int(x_Testudo / z_Testudo)
                                if (np.shape(thresh)[0] > y_Testudo_dash > 0) and (np.shape(thresh)[1] > x_Testudo_dash > 0):
                                    frame[y_Testudo_dash][x_Testudo_dash] = image1[j][k]

        cv2.imshow('Frame', frame)

        # Press a on keyboard to  exit
        if cv2.waitKey(2) & 0xFF == ord('a'):
            break


vid.release()

cv2.destroyAllWindows()