def yolo_to_coords(box, image_size):
    """
    From cx, cy, w, h in Yolov3 format to
    :param box: list of previous tracked detections, initially empty
    :param image_size: list of new detections
    :return: coords
    """
    img_h, img_w = image_size
    # top left
    x1, y1 = int((box[0] - box[2] / 2) * img_w), int((box[1] - box[3] / 2) * img_h)
    # bottom right
    x2, y2 = int((box[0] + box[2] / 2) * img_w), int((box[1] + box[3] / 2) * img_h)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}