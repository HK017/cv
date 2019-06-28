import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, \
    Conv2D, MaxPool2D, \
    LeakyReLU, \
    BatchNormalization, \
    ZeroPadding2D,\
    Add,UpSampling2D,\
    Concatenate
from keras.models import Model
from keras.regularizers import l2


class YOLOV3(object):
    def __init__(self):
        # TODO
        self.classes = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
        self.anchors = [(10,13), (16,30),
                        (33,23), (30,61),
                        (62,45), (59,119),
                        (116,90), (156,198),
                        (373,326)]
        self.num_classes = len(self.classes)
        self.num_anchors = len(self.anchors)/3
        self.decay = 5e-4
        self.learning_rate = 1e-3
        self.images_size = 416, 416, 3
        self.batch_size = 5

    def DarknetConv2D_BN_Leaky(self, x, filters, kernel_size=3, strides=1):
        if strides == 2:
            padding = 'valid'
        else:
            padding = 'same'
        net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=l2(self.decay), use_bias=False)(x)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.1)(net)
        return net

    def residual_block(self, x, num_filters, num_blocks):
        """先补0，再下采样,因为不补0，用默认的same得不到想要的特征图大小"""

        # Downsample
        # ((top_pad, bottom_pad), (left_pad, right_pad))
        net = ZeroPadding2D(((1, 0), (1, 0)))(x)
        net = self.DarknetConv2D_BN_Leaky(net, num_filters, 3, 2)
        for i in range(num_blocks):
            residual_branch = self.DarknetConv2D_BN_Leaky(net, num_filters//2, 1, 1)
            residual_branch = self.DarknetConv2D_BN_Leaky(residual_branch, num_filters, 3, 1)
            net = Add()([net, residual_branch])
        return net

    def darknet53_head(self):
        end_points = {}
        inputs = Input(shape=self.images_size, dtype=tf.float32)
        net = self.DarknetConv2D_BN_Leaky(inputs, 32, 3, 1)
        # residual_blocks  1,2,,8,8,4
        net = self.residual_block(net, 64, 1)
        net = self.residual_block(net, 128, 2)
        net = self.residual_block(net, 256, 8)
        end_points['y31'] = net
        net = self.residual_block(net, 512, 8)
        end_points['y21'] = net
        net = self.residual_block(net, 1024, 4)
        # end one layers
        net = self.DarknetConv2D_BN_Leaky(net, 512, 1, 1)
        net = self.DarknetConv2D_BN_Leaky(net, 1024, 3, 1)
        net = self.DarknetConv2D_BN_Leaky(net, 512, 1, 1)
        net = self.DarknetConv2D_BN_Leaky(net, 1024, 3, 1)
        net = self.DarknetConv2D_BN_Leaky(net, 512, 1, 1)
        end_points['y22'] = net
        # y2
        net = self.DarknetConv2D_BN_Leaky(net, 1024, 3, 1)
        net = Conv2D(self.num_anchors*(self.num_classes+5), 1, 1, kernel_regularizer=l2(self.decay))(net)
        end_points['y1'] = net

        return inputs, end_points

    # TODO
    def make_last_layers(self, x, num_filters, out_filters):
        x = self.DarknetConv2D_BN_Leaky(x, num_filters, 1, 1)
        x = self.DarknetConv2D_BN_Leaky(x, num_filters*2, 3, 1)
        x = self.DarknetConv2D_BN_Leaky(x, num_filters, 1, 1)
        x = self.DarknetConv2D_BN_Leaky(x, num_filters*2, 3, 1)
        x = self.DarknetConv2D_BN_Leaky(x, num_filters, 1, 1)

        return x


    def darknet53_output(self, darknet53_head):
        inputs, end_points = darknet53_head[0], darknet53_head[1]
        # y1
        y1 = end_points['y1']
        # y2
        y21 = end_points['y21']
        y22 = self.DarknetConv2D_BN_Leaky(end_points['y22'], 256, 1, 1)
        y22 = UpSampling2D(2)(y22)
        y2 = Concatenate()([y21, y22])  # 等价于 K.concatenate(inputs, axis=self.axis)
        y2 = self.DarknetConv2D_BN_Leaky(y2, 256, 1, 1)
        y2 = self.DarknetConv2D_BN_Leaky(y2, 512, 3, 1)
        y2 = self.DarknetConv2D_BN_Leaky(y2, 256, 1, 1)
        y2 = self.DarknetConv2D_BN_Leaky(y2, 512, 3, 1)
        y2 = self.DarknetConv2D_BN_Leaky(y2, 256, 1, 1)
        end_points['y32'] = y2
        y2 = self.DarknetConv2D_BN_Leaky(y2, 512, 3, 1)
        y2 = Conv2D(self.num_anchors*(self.num_classes + 5), 1, 1, kernel_regularizer=l2(self.decay))(y2)
        # y3
        y32 = self.DarknetConv2D_BN_Leaky(end_points['y32'], 128, 1, 1)
        y32 = UpSampling2D(2)(y32)
        y3 = Concatenate()([end_points['y31'], y32])
        y3 = self.DarknetConv2D_BN_Leaky(y3, 128, 1, 1)
        y3 = self.DarknetConv2D_BN_Leaky(y3, 256, 3, 1)
        y3 = self.DarknetConv2D_BN_Leaky(y3, 128, 1, 1)
        y3 = self.DarknetConv2D_BN_Leaky(y3, 256, 3, 1)
        y3 = self.DarknetConv2D_BN_Leaky(y3, 128, 1, 1)
        y3 = self.DarknetConv2D_BN_Leaky(y3, 256, 3, 1)
        y3 = Conv2D(self.num_anchors*(self.num_classes + 5), 1, 1, kernel_regularizer=l2(self.decay))(y3)
        model = Model(inputs, [y1, y2, y3])

        return model

    def decode(self, feats, is_traing=True):
        """

        :param feats: yolov3 output
        :param is_traing: if training
        :return:
        """
        anchors_tensor = K.reshape(self.anchors, (1, 1, 1, self.num_anchors, 2))
        grid_shape = K.shape(feats)[1:3] # height, width
        grid_y = K.tile(K.reshape(K.arange(0, grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[0], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, grid_shape[1]), [1, -1, 1, 1]),
                        [1, grid_shape[1], 1, 1])
        grid = K.concatenate([grid_x, grid_y], axis=-1)
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], self.num_anchors, self.num_classes + 5])

        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape, K.dtype(feats))
        box_hw = K.exp(feats[..., 2:4]) / K.cast(grid_shape[:2], K.dtype(feats)) * anchors_tensor
        box_confidence = K.sigmoid(feats[..., 4])
        box_class_probs = K.sigmoid(feats[..., 5:])
        if is_traing:

            return grid, feats, box_xy, box_hw, box_confidence, box_class_probs

        return


    def preprocess_gt_boxes(self, true_boxes):
        num_layers = self.num_anchors // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.zeros_like(true_boxes, dtype=K.dtype(true_boxes))
        box_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4])//2
        box_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = box_xy / self.images_size[0:2][::-1]
        true_boxes[..., 2:4] = box_wh / self.images_size[0:2][::-1]

        # gain features map size
        grid_shape = [(13, 13), (26, 26), (52, 52)]

        # list len=3
        y_true = [np.zeros((self.batch_size, grid_shape[l][0], grid_shape[l][1],
                            self.num_anchors, self.num_classes + 5), dtype=np.float32)
                  for l in range(3)]

        for j in range(self.batch_size):
            for i in range(3):
                y_true[i][j, ]


    def box_iou(self, box1, box2):
        """
        calc iou with gt_boxes and yuce boxes
        :param box1: [batch, n, n, num_anchors, 4], 4 is [x_ct,y_ct, w, h]
        :param box2: gt_boxes -- > [batch, n, n, 5], 4 is [x_ct,y_ct, w, h, ]
        :return: iou shape is [batch , n, n, ]
        """

        gt_boxes_maxs = box2[..., 0:2] + box2[..., 2:4] / 2
        gt_boxes_mins = box2[..., 0:2] - box2[..., 2:4] / 2

        box1_maxs = box1[..., 0:2] + box1[..., 2:4] / 2
        box1_mins = box1[..., 0:2] - box1[..., 2:4] / 2

        lu = np.maximum(box1_mins, gt_boxes_mins)
        rd = np.minimum(box1_maxs, gt_boxes_maxs)
        intersection = np.maximum(rd - lu, 0.)
        inter_area = intersection[..., 0] * intersection[..., 1]
        union_area1 = (box1_maxs[..., 0] - box1_mins[..., 0])*(box1_maxs[..., 1] - box1_mins[..., 1])
        union_area2 = (gt_boxes_maxs[..., 0] - gt_boxes_mins[..., 0])*(gt_boxes_maxs[..., 1] - gt_boxes_mins[..., 1])
        ious = inter_area / (union_area1 + union_area2 - inter_area)

        return ious


if __name__ == '__main__':
    yolo = YOLOV3()
    yolo_output = yolo.darknet53_output(yolo.darknet53_head())
    # print(yolo_output.output_shape)
    grid_x, grid_y = yolo.decode()
    print(grid_x.get_shape())