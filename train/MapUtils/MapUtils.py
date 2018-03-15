#Daniel D. Lee, Alex Kushleyev, Kelsey Saulnier, Nikolay Atanasov
import numpy as np

# INPUT 
# im              the map 
# x_im,y_im       physical x,y positions of the grid map cells
# vp(0:2,:)       occupied x,y positions from range sensor (in physical unit)  
# xs,ys           physical x,y,positions you want to evaluate "correlation" 
#
# OUTPUT 
# c               sum of the cell values of all the positions hit by range sensor
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


#Bresenham's line algorithm
def getMapCellsFromRay(x0t,y0t,xis,yis):
	nPoints = np.size(xis)
	xyio = np.array([[],[]])
	for x1,y1  in zip(xis,yis):
		x0 = x0t
		y0 = y0t
		steep = (np.abs(y1-y0) > np.abs(x1-x0))
		if steep:
			temp = x0
			x0 = y0
			y0 = temp
			temp = x1
			x1 = y1
			y1 = temp
		if x0 > x1:
			temp = x0
			x0 = x1
			x1 = temp
			temp = y0
			y0 = y1
			y1 = temp
		deltax = x1 - x0
		deltay = np.abs(y1 - y0)
		error = deltax / 2.
		y = y0
		ystep = 0
		if y0 < y1:
		  ystep = 1
		else:
		  ystep = -1
		if steep:
			for x in np.arange(x0,x1):
				xyio = np.concatenate((xyio,np.array([[y],[x]])),axis=1)
				error = error - deltay
				if error < 0:
					y += ystep
					error += deltax
		else:
			for x in np.arange(x0,x1):
				xyio = np.concatenate((xyio,np.array([[x],[y]])),axis=1)
				error = error - deltay
				if error < 0:
					y += ystep
					error += deltax
	return xyio


