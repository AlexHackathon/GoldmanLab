import colorsys
import matplotlib.pyplot as plt
import numpy as np


def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    #return (int(255 * r), int(255 * g), int(255 * b))
    return (r, g, b)

def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]
