import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def nll_loss(output, target):
    return F.nll_loss(output, target)


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if is_cuda:
        v = v.cuda()
    return v


def forward(im_data, gt_boxes=None, gt_classes=None, dontcare=None, size_index=0):
    conv1s = self.conv1s(im_data)
    conv2 = self.conv2(conv1s)
    conv3 = self.conv3(conv2)
    conv1s_reorg = self.reorg(conv1s)
    cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
    conv4 = self.conv4(cat_1_3)
    conv5 = self.conv5(conv4)  # batch_size, out_channels, h, w
    global_average_pool = self.global_average_pool(conv5)

    # for detection
    # bsize, c, h, w -> bsize, h, w, c ->
    #                   bsize, h x w, num_anchors, 5+num_classes
    bsize, _, h, w = global_average_pool.size()
    global_average_pool_reshaped = \
        global_average_pool.permute(0, 2, 3, 1).contiguous().view(bsize,
                                                                  -1, cfg.num_anchors, cfg.num_classes + 5)  # noqa

    # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
    xy_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
    wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
    bbox_pred = torch.cat([xy_pred, wh_pred], 3)
    iou_pred = F.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])

    score_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()
    prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)  # noqa

    bbox_pred_np = bbox_pred.data.cpu().numpy()
    iou_pred_np = iou_pred.data.cpu().numpy()
    _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = _build_target(bbox_pred_np,
                                                                               gt_boxes,
                                                                               gt_classes,
                                                                               dontcare,
                                                                               iou_pred_np,
                                                                               size_index)

    _boxes = Variable(torch.from_numpy(np_to_variable(_boxes)), type(torch.FloatTensor), volatile=False)
    _ious = Variable(torch.from_numpy(np_to_variable(_ious)), type(torch.FloatTensor), volatile=False)
    _classes = Variable(torch.from_numpy(np_to_variable(_classes)), type(torch.FloatTensor), volatile=False)
    box_mask = Variable(torch.from_numpy(np_to_variable(_box_mask)), type(torch.FloatTensor), volatile=False)
    iou_mask = Variable(torch.from_numpy(np_to_variable(_iou_mask)), type(torch.FloatTensor), volatile=False)
    class_mask = Variable(torch.from_numpy(np_to_variable(_class_mask, type(torch.FloatTensor), volatile=False)

    num_boxes = sum((len(boxes) for boxes in gt_boxes))

    # _boxes[:, :, :, 2:4] = torch.log(_boxes[:, :, :, 2:4])
    box_mask = box_mask.expand_as(_boxes)

    self.bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes  # noqa
    self.iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes  # noqa

    class_mask = class_mask.expand_as(prob_pred)
    self.cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask,
                                                   _classes * class_mask) / num_boxes  # noqa

    return bbox_pred, iou_pred, prob_pred

