import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_images(img1, img2, title1, title2):
    # Display two images side by side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.show()


def detect_rooms(img, background_label, spaces_label):

    # Create a binary mask
    binary_mask = (img < 255).astype(np.uint8) * 255

    # Erode the image
    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(binary_mask, kernel, iterations=2)

    # Detect edges in the eroded image
    edges = cv2.Canny(eroded_img, 50, 150)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=1, maxLineGap=300)

    # Create a blank canvas to draw lines
    connected_lines = np.zeros_like(eroded_img)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(connected_lines, (x1, y1), (x2, y2), 255, thickness=2)

    # Combine the original eroded image with the connected lines
    final_img = ~(cv2.bitwise_or(eroded_img, connected_lines)) ## here we invert the image to use spaces as foreground

    # plt.imshow(~final_img, cmap='gray')
    # plt.title("Final Image (inverted)")
    # plt.show()

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_img, connectivity=8)

    # print('stats:', stats)

    # Filter components by area and create a colored output image
    min_area = 20000  # Minimum area threshold
    output_img = np.zeros((*final_img.shape, 3), dtype=np.uint8)  # Create a blank color image

    # Find the largest component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background (label 0)

    for label in range(1, num_labels):  # Skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if label == largest_label:
            output_img[labels == label] = background_label  # Set/assume the largest component to background_label
        elif area >= min_area:
            # Generate a random color for the component
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            output_img[labels == label] = color
        else:
            output_img[labels == label] = spaces_label  # Set small components to white

    return output_img



spaces_label = 150
background_label = 200

if __name__ == '__main__':

    # Load image (grayscale)
    img = cv2.imread("./data/floor_plan.png", cv2.IMREAD_GRAYSCALE)
    # print('shape:', img.shape)

    # Segment rooms
    rooms_mask = detect_rooms(img, background_label=background_label, spaces_label=spaces_label)
    # plt.imshow(rooms_mask)
    # plt.title('detected rooms')
    # plt.show()


    # Find connected components in the combined mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rooms_mask[:, :, 0], connectivity=8)
    #
    # print('labels', labels)
    # print('num_labels', num_labels)

    # Create a copy for visualization
    output_img = cv2.cvtColor(rooms_mask[:, :, 0], cv2.COLOR_GRAY2BGR)

    # Iterate over each connected component
    for label in range(0, num_labels):  # Skip background (label 0)
        # Get the bounding box and area for the component
        x, y, w, h, area = stats[label]

        # Get the original value of the component in the combined mask
        original_value = rooms_mask[:, :, 0][y:y + h, x:x + w][labels[y:y + h, x:x + w] == label][0]

        # print('original_value', original_value)

        if original_value == 150:
            segment_type = 'window/door'
        elif original_value == 0:
            segment_type = 'floor'
        else:
            segment_type = 'room'

        # Prepare text with area, width, and height
        text = f"A:{area}, W:{w}, H:{h}, L:{label}, {segment_type}"

        # Get the centroid of the component
        cx, cy = centroids[label]

        # set color
        if original_value == 0:
            color = (0, 255, 0)
            cx = cx - cx // 2
            cy = cy - cy // 2
        else:
            color = (0, 0, 255)

        # Add the text to the image at the centroid
        cv2.putText(output_img, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv2.LINE_AA)


    # Display results
    visualize_images(img, output_img, 'original plan', 'Segmented Rooms')

