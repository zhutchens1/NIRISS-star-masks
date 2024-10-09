import numpy as np

def dist_from_cent(xx,yy):
    return np.sqrt((xx-1500)**2. + (yy-1500)**2.)

data = np.array([
    [18.78, 1800, 1607],
    [16.91, 1868, 1172],
    [16.82, 1782, 1240],
    [20.44, 1459, 1455],
    [20.35, 1489, 1439],
    [19.42, 1463, 1316],
    [21.08, 1412, 1467],
    [16.36, 1427, 1099],
    [16.31, 1647, 2232],
    [17.42, 1403, 1025],
    [14.60, 1195, 6.70],
    [12.00, 2.00, 2840]
])

gband = data[:,0]
radii = dist_from_cent(data[:,1], data[:,2])

popt = np.polyfit(gband, radii, 1)
def scaledfit(gmag):
    xx=np.array((np.polyval(popt,gmag) + (np.max(radii)-np.polyval(popt, gband[np.argmax(radii)]))))
    cond = (xx>20).astype(int)
    return cond*xx + (1-cond)*20

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(gband, radii, 'ko')
    plt.plot(gband, np.polyval(popt, gband), 'r-', label='true fit')
    plt.plot(gband, scaledfit(gband), 'g-', label='calibrated radii for masking')
    plt.legend(loc='best')
    plt.show()

