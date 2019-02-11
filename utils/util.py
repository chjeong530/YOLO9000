import os
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def non_max_suppresssion_fast(boxes, probs, threshold=0.5):

    '''

    :param boxes:
    :param probs:
    :param threshold:
    :return:

    1. 주어진 Box들 중 가장 높은 Score를 가진 Box 선택
    2. 선택된 Box와 나머지 Box들 간의 IOU를 계산하고 threshold 이상이면 제거(동일한 객체에 대한 검출이기 때문에 겹치는 부분이 많을 것이라고 예상됨.)
    3. 특정한 Box의 숫자가 남을 때까지 or 더 이상 선택할 Box가 없을 때 까지 위의 과정 반복
    '''

    # Init the picked box info
    pick = []

    # Box coordinate
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute area of each boxes
    area = (x2 - x1) * (y2 - y1)

    # Sort
    idxs = np.argsort(probs)

    while idxs.size:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        iou = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > threshold)[0])))

    return boxes[pick].astype("int")

