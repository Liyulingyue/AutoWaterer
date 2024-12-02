def find_closest_point_on_line(x1, y1, x2, y2, target_x, target_y):
    if x1 == x2 and y1 == y2:
        return x1, y1  # 如果点1和点2重合，直接返回该点

    # 计算直线斜率 m
    if x2 - x1 == 0:
        m = None  # 垂直线
    else:
        m = (y2 - y1) / (x2 - x1)

    # 计算截距 b
    if m is not None:
        b = y1 - m * x1
    else:
        b = x1  # 垂直线时，b为x1

    # 如果直线是垂直的
    if m is None:
        closest_x = x1
        closest_y = target_y
    else:
        # 计算目标点到直线的垂直投影
        closest_x = (m * target_y - m * b + target_x) / (m ** 2 + 1)
        closest_y = m * closest_x + b

    return closest_x, closest_y


def calculate_iou(box1, box2):
    """
    计算两个边界框的IOU（重合度）。

    参数:
    box1, box2: 列表，格式为 [xmin, ymin, xmax, ymax]

    返回:
    IOU值，范围为0到1之间。
    """

    # 计算box1和box2的交集坐标
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # 计算交集区域的宽度和高度
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)

    # 计算交集面积
    inter_area = inter_width * inter_height

    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算IOU
    iou = inter_area / float(min(box1_area, box2_area))

    return iou