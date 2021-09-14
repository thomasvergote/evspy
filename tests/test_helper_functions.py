from evspy.utils.helper_functions import semilogintersection, loglogintersection, line_intersection
import numpy as np

def test_line_intersection():
    line1 = ((0.5,1),(1.5,0.5))
    line2 = ((1,0),(2,2))
    x,y = line_intersection(line1,line2)
    assert x == 1.3
    assert y == 0.6

def test_semilogintersection():
    e0 = 2
    e1 = 1
    sigma1 = 1
    Cc = 1
    Cr = 0
    x,y = semilogintersection(e0, e1, sigma1, Cc, Cr)
    assert x == 10
    assert y == 1

def test_loglogintersection():
    e0 = 2
    e1 = 1
    sigma1 = 1
    rhoc = 1
    Cr = 0
    x = loglogintersection(e0, e1, sigma1, rhoc, Cr)
    assert np.round(x[0],1) == 2