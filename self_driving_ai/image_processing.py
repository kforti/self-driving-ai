import numpy as np
import cv2
from self_driving_ai.draw_lanes import draw_lanes
from skimage.io import imread, imsave

def roi(img, vertices):
    # blank mask:
    mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, 255)

    # returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)

    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    imsave("processed_img.jpeg", processed_img)

    vertices = np.array([[0, 140], [25, 140], [60, 140], [100, 140], [200, 140], [260, 140],], np.int32)

    #processed_img = roi(processed_img, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, 20, 15)
    m1 = 0
    m2 = 0
    try:
        l1, l2, m1, m2 = draw_lanes(original_image, lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)


            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img, original_image, m1, m2

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img_path = "/home/kevin/Documents/IMG/center_2020_05_13_10_53_05_307.jpg" #"/home/kevin/Documents/IMG/center_2020_05_13_10_52_55_441.jpg"
    img = imread(img_path)
    plt.imshow(img)
    plt.show()
    new_screen, original_image, m1, m2 = process_img(img)
    imsave("processed_img.jpeg", original_image)
    print(m1)
    print(m2)

