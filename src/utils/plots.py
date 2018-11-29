from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg

def plot_estim(img, estim, target):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    # display img
    ax.imshow(img)

    # add estim and target
    ax.text(0.5, 0.1, f"trg:{target}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r', fontsize=20)
    ax.text(0.5, 0.04, f"est:{estim}", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='r', fontsize=20)

    ser_fig = serialize_fig(fig)
    plt.close(fig)
    return ser_fig

def serialize_fig(fig):
    canvas = plt_backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    return image_chw