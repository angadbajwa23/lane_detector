import cv2
import numpy as np

imgpath="C:\\Users\\Angad Bajwa\\Downloads\\road_2.jpeg"
vidpath="C:\\Users\\Angad Bajwa\\Downloads\\challenge.mp4"
img=cv2.imread(imgpath)
cv2.imshow('input',img)
cap=cv2.VideoCapture(vidpath)
def process_image(img):
    #cvt to grayscale
    grayscaled=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    gaussianBlur = cv2.GaussianBlur(grayscaled, (5, 5), 0)


    # canny
    minThreshold = 100
    maxThreshold = 200
    edgeDetectedImage = cv2.Canny(gaussianBlur, minThreshold, maxThreshold)
    #cv2.imshow('canny',edgeDetectedImage)

    def region_of_interest(img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with
        # depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    # apply mask
    lowerLeftPoint = [130, 540]
    upperLeftPoint = [410, 350]
    upperRightPoint = [570, 350]
    lowerRightPoint = [915, 540]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
                     lowerRightPoint]], dtype=np.int32)
    masked_image = region_of_interest(edgeDetectedImage, pts)
    cv2.imshow('hough', masked_image)

    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

        draw_lines(line_img, lines)
        return line_img

    def draw_lines(img, lines, color=[0, 0, 255], thickness=20):
        """
        This function draws `lines` with `color` and `thickness`.
        """
        imshape = img.shape

        # these variables represent the y-axis coordinates to which
        # the line will be extrapolated to
        ymin_global = img.shape[0]
        ymax_global = img.shape[0]

        # left lane line variables
        all_left_grad = []
        all_left_y = []
        all_left_x = []

        # right lane line variables
        all_right_grad = []
        all_right_y = []
        all_right_x = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
                ymin_global = min(min(y1, y2), ymin_global)

                if (gradient > 0):
                    all_left_grad += [gradient]
                    all_left_y += [y1, y2]
                    all_left_x += [x1, x2]
                else:
                    all_right_grad += [gradient]
                    all_right_y += [y1, y2]
                    all_right_x += [x1, x2]

        left_mean_grad = np.mean(all_left_grad)
        left_y_mean = np.mean(all_left_y)
        left_x_mean = np.mean(all_left_x)
        left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

        right_mean_grad = np.mean(all_right_grad)
        right_y_mean = np.mean(all_right_y)
        right_x_mean = np.mean(all_right_x)
        right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

        # Make sure we have some points in each lane line category
        if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
            upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
            lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
            upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
            lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

            cv2.line(img, (upper_left_x, ymin_global),
                     (lower_left_x, ymax_global), color, thickness)
            cv2.line(img, (upper_right_x, ymin_global),
                     (lower_right_x, ymax_global), color, thickness)

    # hough lines
    rho = 1
    theta = np.pi / 180
    threshold = 30
    min_line_len = 20
    max_line_gap = 20

    houged = hough_lines(masked_image, rho, theta,
                         threshold, min_line_len, max_line_gap)
    cv2.imshow('houghed', houged)

    def weighted_img(img, initial_img, α=1., β=1.3, λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)

    # outline the input image
    colored_image = weighted_img(houged, img)
    return colored_image
img2=process_image(img)
cv2.imshow('output',img2)
cv2.waitKey(0)
