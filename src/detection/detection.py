import torch
import torchvision
import torch.nn as nn


class ROI_pooling(nn.Module):
    """docstring for ROI_pooling
    Assumes: roi is defined by center cordinate along with width and height
    in image reference
    """
    def __init__(self, pool_size=7, sampling_ratio = 16):
        self.pool_size = pool_size
        self.subsampling_ratio = sampling_ratio
        super(ROI_pooling, self).__init__()

    def get_output_shape(self, input_shape):
        feature_map_shp, rois_shape = input_shape
        batch_size = feature_map_shp[0]
        num_rois = rois_shape[1]
        channesls = feature_map_shp[1]
        return (batch_size, num_rois, self.pool_size, self.pool_size, channesls)

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi): 
          return ROIPoolingLayer._pool_roi(feature_map, roi, 
                                           pooled_height, pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas


    @staticmethod
    def _pool_roi(self, feature_map, roi):
        """Applies ROI pooling to single image and single roi"""
        subsampling = self.subsampling_ratio

        x_cnt = int(roi[1]/subsampling)
        y_cnt = int(roi[0]/subsampling)
        width = int(roi[2]/subsampling)
        height = int(roi[3]/subsampling)

        region = feature_map[x_cnt-int(width/2):x_cnt+int(width/2), 
                    y_cnt-int(height/2):y_cnt+int(height/2)]

        
        # w_step = int(width/self.pool_size)
        # h_step = int(height/self.pool_size)

        # areas = [[(
        #             i*h_step, 
        #             j*w_step, 
        #             (i+1)*h_step if i+1 < pooled_height else region_height, 
        #             (j+1)*w_step if j+1 < pooled_width else region_width
        #            ) 
        #            for j in range(pooled_width)] 
        #           for i in range(pooled_height)]
        
        # # take the maximum of each area and stack the result
        # def pool_area(x): 
        #   return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
        
        # pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])

        resize_region = nn.functional.upsample(region, size=(14,14), mode='bilinear')
        pool_layer = nn.MaxPool2d(2)
        pooled_feature = pool_layer(resize_region)

        return pooled_feature






    def forward(feature_map, proposals):
        
        return ROI_pooling
        

class Detector(nn.Module):
    """docstring for Detector"""
    def __init__(self, input_size, hidden_size, class_size):
        super(Detector, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.class_out = nn.Sequential(nn.Linear(hidden_size, class_size), nn.Softmax())
        self.bbox = nn.Linear(hidden_size, class_size)
        
    def forward(rois):
        out = self.fc1(rois)
        classes = self.class_out(out)
        bbox_out = self.bbox(out)

        return classes, bbox_out