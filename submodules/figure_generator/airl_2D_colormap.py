import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio


class AIRL_ColorMap(object):
    @classmethod
    def get_2D_colormap(cls, data: np.ndarray, angle_offset=0, rank_based_coloring=True, center=None, numpy_max=None, max_l=None):

        if rank_based_coloring:
            arg_sort = np.argsort(data, axis=0)
            res = np.zeros_like(data)
            for i in range(np.size(data, 0)):
                for j in range(np.size(data, 1)):
                    res[arg_sort[i, j], j] = i
            data = res

        data = data.reshape((-1, 2))
        N = np.size(data, axis=0)
        minXY = np.min(data, axis=0)
        maxXY = np.max(data, axis=0)
        # center = (maxXY + minXY) / 2
        # print("center", np.median(data, axis=0), center)
        if center is None:
            center = np.median(data, axis=0)

        # center data
        data = data - center

        # "normalise data"
        # print("max", np.max(np.abs(data), axis=0), numpy_max)
        if numpy_max is None:
            data = data / np.max(np.abs(data), axis=0)
        else:
            data = data / numpy_max

        angle = np.arctan2(data[:,1], data[:,0]) + angle_offset
        angle = angle.reshape((-1, 1))
        print(np.min(angle))

        l = np.cbrt(np.sum(data ** 2, axis=1))
        if max_l is None:
            l = l / np.max(l)
        else:
            l = l / max_l
        # print("max_l", np.max(l), max_l)
        l = l.reshape((-1, 1))

        C = cls.get_color_function(angle)

        C = C * 0.7 + 0.3
        C = l * C + (1. - l) * 0.75
        return C

    @classmethod
    def get_color_function(cls, t: np.ndarray) -> np.ndarray:
        t = t.flatten()
        r = (np.sin(t + 1.3 * np.pi) / 2. + 0.5) * 0.8 + 0.1
        g = (np.sin(t) / 2.) * 0.6 + 0.6
        b = 1. - ((np.sin(t + 1.6 * np.pi) / 2.) + 0.5)

        return np.vstack((r, g, b)).T


if __name__ == '__main__':

    N = 2500
    data = np.random.rand(2 * N).reshape((-1, 2))
    print(AIRL_ColorMap.get_2D_colormap(data))
    plt.scatter(data[:,0], data[:,1], c=AIRL_ColorMap.get_2D_colormap(data))
    plt.show()

    colormap = AIRL_ColorMap.get_2D_colormap(data)
    trace = dict(type='scatter',
                 x=data[:,0],
                 y=data[:,1],
                 mode='markers',
                 marker=dict(color=[
                     f'rgb({colormap[i, 0]}, {colormap[i, 1]}, {colormap[i, 2]})' for i
                     in range(data.shape[0])], size=10)
                 )

    pio.show(trace)


