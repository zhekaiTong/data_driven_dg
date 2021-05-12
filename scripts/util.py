import cv2

def draw_circular_mask(image, center, radius):
    """Draw a circle at the center of the given image."""
    #h, w = image.shape[:2]
    #print(h,w)
    #center = (w // 2, h // 2)

    # Evaluate radius
    inner_radius = radius
    cv2.circle(image, center, inner_radius, (255, 255, 255), -1)
    return image
