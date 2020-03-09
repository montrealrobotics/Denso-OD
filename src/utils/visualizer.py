import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches
from .projection import ground_project


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 ).reshape((h,w,3))
    # buf.shape = ( w, h,4 )
    #
    # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    # buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

class Visualizer(object):
    """docstring for Visualizer."""

    def __init__(self, img, instances, img_path, rpn_proposals, cfg):
        super(Visualizer, self).__init__()
        self.image = img
        self.proposals = rpn_proposals
        self.instances = instances
        self.path = img_path
        self.output = plt.figure(figsize=(16, 12), dpi=80)
        self.cfg = cfg

    def draw_instances(self):
        class_labels = self.cfg.INPUT.LABELS_TO_TRAIN

        img = self.image.copy()
        drawer = ImageDraw.Draw(img, mode=None)

        for instance in self.instances:
            box = instance.pred_boxes
            drawer.rectangle(box, outline ='red' ,width=3)
            if instance.has("pred_classes"):
                drawer.text([box[0], box[1]-10],"{}: {:.2f}%".format(class_labels[instance.pred_classes],
                instance.scores), outline='green')

            if instance.has("pred_variance"):
                sigma = np.sqrt(instance.pred_variance)
                drawer.ellipse([box[0]-2*sigma[0], box[1]-2*sigma[1], box[0]+2*sigma[0], box[1]+2*sigma[1]], outline='blue', width=3)
                drawer.ellipse([box[2]-2*sigma[2], box[3]-2*sigma[3], box[2]+2*sigma[2], box[3]+2*sigma[3]], outline='blue', width=3)
        ax = self.output.add_subplot(1,2,1)
        ax.imshow(img)

        # return np.asarray(self.image)
        return ax

    def draw_instance_prob(self):
        
        img = self.image.copy()

        ax = self.output.add_subplot(1,2,1)
        ax.imshow(img)

        for instance in self.instances:
            box_cords = instance.pred_boxes
            box = patches.Rectangle(box_cords[[0,3]], box_cords[2]-box_cords[0], box_cords[3]-box_cords[1], linewidth=1, fill=False, edgecolor='r')
            ax.add_patch(box)
        
        return ax


    def draw_proposals(self):
        for instance in self.instances:
            box = instance.proposal_boxes.tensor.cpu().numpy()[0]
            drawer.rectangle(box, outline ='red' ,width=3)

    def draw_projection(self):
        xy_coords, variance = ground_project(self.instances, self.path)
        ax = self.output.add_subplot(1,2,2)

        for xy, xy_var in zip(xy_coords, variance):
            ellip = patches.Ellipse(xy, width= 4*xy_var[0], height= 4*xy_var[1], fill = False)
            box_width, box_height = 2, 4
            box = patches.Rectangle((xy[0]-box_width/2, xy[1]), box_width, box_height, linewidth=1, fill=False, edgecolor='r')
            ellip.set_edgecolor(np.random.rand(3))
            ax.add_patch(ellip)
            ax.add_patch(box)

        # ax.axis('image')
        # ax.axis('auto')
        ax.set_xlim(-15.0, 15.0)
        ax.set_ylim(0.0, 50.0)
        ax.set_aspect('equal')

        return ax

    def get_image(self):
        return fig2data(self.output)

    def save(self, direc):
        # plt.show()
        plt.savefig(direc+self.path[-11:-4]+".png")
        plt.close()

    def show(self):
        # plt.show(block=False)
        # plt.pause(3)
        # plt.close()

        plt.draw()
        # plt.pause(3)
        plt.waitforbuttonpress(0)
        plt.close()