
def area(tl_x,tl_y,br_x,br_y):
    """ Calculates Area of box
    Returns:
    area (number)
    """
    return (br_y-tl_y) * (br_x-tl_x)

def IoU(box1, box2):
    """ The intersection over union (IoU) between box1 and box2
    Arguments:
    box1 - first box, list object with coordinates [tl_x, tl_y, w, h] # tl-top left, br-bottom right
    box2 - second box, list object with coordinates [tl_x, tl_y, w, h] # tl-top left, br-bottom right
    Returns:
    iou - IoU (number)
    """
    # Get boxes bottom right corners
    box1_br_x = box1[0] + box1[2] # tl_x + w
    box1_br_y = box1[1] + box1[3] # tl_y + h
    box2_br_x = box2[0] + box2[2]
    box2_br_y = box2[1] + box2[3]
    # Find Intersection (tl_y, tl_x, br_y, br_x) of box1 and box2
    inter_tl_x = max(box1[0], box2[0]) # top left corner
    inter_tl_y = max(box1[1], box2[1])
    inter_br_x = min(box1_br_x, box2_br_x)
    inter_br_y = min(box1_br_y, box2_br_y)

    # If doesn't overlap return 0
    if inter_br_x < inter_tl_x or inter_br_y < inter_tl_y:
        return 0.0

    # Calculate Intersection Area
    inter_area = area(inter_tl_x,inter_tl_y,inter_br_x,inter_br_y)

    # Calculate the Union area: Union(A,B) = A + B - Inter(A,B)
    box1_area = area(box1[0],box1[1],box1_br_x,box1_br_y)
    box2_area = area(box2[0],box2[1],box2_br_x,box2_br_y)
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area
    return iou

def xyxy2xywh(b):
    x = b[0]
    y = b[1]
    w = b[2]-b[0]
    h = b[3]-b[1]
    return [x,y,w,h]
