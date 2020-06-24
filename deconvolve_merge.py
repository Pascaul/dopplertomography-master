import numpy as np
from tables import *
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits as pyfits 
import math
from scipy.interpolate import splrep, splev
from astropy.stats import sigma_clip
import numpy.polynomial.polynomial as poly
import scipy.signal
from scipy import signal,interpolate
import argparse
from scipy.optimize import curve_fit

def fit_splines(x,y,xdata,nknots):

	# Luego el arreglo de puntos: 
	knots = np.arange(x[1],x[len(x)-1],(x[len(x)-1]-x[1])/np.double(nknots)) 
	idx_knots = (np.arange(1,len(x)-1,(len(x)-2)/np.double(nknots))).astype('int') 
	knots = x[idx_knots] 

	tck = splrep(x,y,t=knots) 
	fit = splev(xdata,tck) 
	return fit

def clean_lines(wav,flux,spectype):
	if spectype == 'A':	####A or B type 
		mask1 = ((wav >6550)&(wav <6584))
		mask2 = ((wav>5875)&(wav<5884))
		mask3 = ((wav>4333)&(wav<4353))
		mask4 = ((wav>4850)&(wav<4870))
		mask5 = ((wav>4090)&(wav<4110))
		#mask6 = ((wav>3875)&(wav<3900))
		#mask7 = ((wav>3920)&(wav<3945))
		#mask8 = ((wav>3955)&(wav<3980))
		mask9 = (wav<3850)
		mask10 = ((wav>4854)&(wav<4870))
		mask11 = ((wav>4470)&(wav<4486))
		mask12 = ((wav>4336)&(wav<4360))
		mask = mask1+mask2+mask3+mask4+mask5+mask9+mask10+mask11+mask12
		#mask = mask6 + mask7 + mask8 
		
	
	elif spectype == 'F':	#### F type

		mask1 = ((wav > 6552) & (wav < 6572))
		mask2 = ((wav > 6488) & (wav < 6500))
		mask3 = ((wav > 5887) & (wav < 5900))
		mask4 = ((wav > 4845) & (wav < 4878))
		#mask5 = ((wav > 4470) & (wav < 4486))
		mask6 = ((wav > 4336) & (wav < 4360))
		mask7 = ((wav > 4333) & (wav < 4353))
		#mask8 = ((wav > 4090) & (wav < 4110))
		mask9 = ((wav > 3963) & (wav < 3975))
		#mask10 = ((wav > 3875) & (wav < 3900))
		#mask11 = ((wav > 3955) & (wav < 3980))
		mask12 = ((wav > 3925) & (wav < 3941))
		mask13 = (wav < 3850)

		mask = mask1+mask2+mask4+mask7+mask12+mask13
		#mask = mask + mask5 + mask8 + mask10 + mask11

	return ~mask

c =299792.458 #kms

def n_Edlen(l):
    sigma = 1e4 / l
    sigma2 = sigma*sigma
    n = 1 + 1e-8 * (8342.13 + 2406030 / (130-sigma2) + 15997/(38.9-sigma2))
    return n

def ToAir(l):
    return (l / n_Edlen(l))

def ToVacuum(l):
    cond = 1
    l_prev = l.copy()
    while(cond):
        l_new = n_Edlen(l_prev) * l
        if (max(np.absolute(l_new - l_prev)) < 1e-10):
            cond = 0
        l_prev = l_new
    return l_prev

def shift_doppler(obs_wav,vel):
	return obs_wav/(vel*1e13/(c*1e13)+1)

def addcolsbyrow(array1,array2):
    auxtable = Table([[],[] ], names=('time','flux'),dtype=(float,float))
    for i in np.arange(len(array1)):
        table = [array1[i], array2[i]]
        auxtable.add_row(table)
    return auxtable

def ordenar(array1,array2):
    zp = list(zip(array1,array2))
    sort = sorted(zp)
    array1,array2 = zip(*sort)
    array1 = np.array(array1)
    array2 = np.array(array2)

    mask = np.isnan(array2)
    array1 = array1[~mask]
    array2 = array2[~mask]
    
    mask = np.isnan(array1)
    array1 = array1[~mask]
    array2 = array2[~mask]
    

    return array1,array2

###Tomar un orden aproximarlo en los bordes, hacerle mascara para generar orden nuevo
def cont(f):
	# this function performs a continuum normalization
	x = np.arange(len(f))
	fo = f.copy()
	xo = x.copy()
	mf = scipy.signal.medfilt(f,31)
	c = np.polyfit(x,mf,3)

	while True:
		m = np.polyval(c,x)
		res = f - m
		I = np.where(res>0)[0]
		dev = np.median(res[I])
		J = np.where((res>-1*dev)&(res<4*dev))[0]
		#print len(J)
		K = np.where(res<-1*dev)[0]
		H = np.where(res>4*dev)[0]
		if (len(K)==0 and len(H) == 0) or len(J)<0.5*len(fo):
			break
		x,f = x[J],f[J]
		c = np.polyfit(x,f,3)
	return np.polyval(c,xo)


def rlineal(im,swav,sflux):
	lineal_flux = []
	lineal_wav = []

	for i in np.arange(len(im[0,:,0])-3):
		wav = im[0,i+3,:]
		flux = im[1,i+3,:]

		#plt.plot(wav,flux)
		#plt.plot(wav,flux/cont(flux))
		#plt.plot(wav,cont(flux))
		#plt.show()
		#raise
		auxswav = swav.copy()
		auxsflux = sflux.copy()	
		#plt.plot(auxswav,auxsflux)
		#plt.show()
		mask = ( auxswav < np.max(wav)) & (auxswav > np.min(wav) )
		auxswav = auxswav[mask]
		auxsflux = auxsflux[mask]

		#plt.plot(auxswav,auxsflux)
		#plt.show()
		#raise
		
		nflux = flux/cont(flux)

		tck = interpolate.splrep(auxswav,auxsflux,k=3)

		iflux = interpolate.splev(wav,tck)

		auxsflux /= np.add.reduce(auxsflux)
		nflux /= np.add.reduce(nflux)
		iflux /= np.add.reduce(iflux)
		
		dif = np.median(auxsflux) - np.median(iflux)
		lineal_wav.append(np.array(wav))
		lineal_flux.append(np.array(iflux)) 

		###
		#plt.plot(auxswav,auxsflux-dif)
		#plt.show()
		#plt.plot(wav,iflux,'o')
		#plt.show()

	#plt.show()
	return lineal_wav,lineal_flux


def linealizacion(im,linstep,mode):
	lineal_x = []
	lineal_y = []
	for i in np.arange(len((im[0,:,0]))):
		print 'max', np.max(im[0,i,:])
		print len (im[0,i,:])
		if np.max(im[0,i,:]) <= 5500:
			print 'esta condicion'
			step = 3201
			auxx = im[0,i,:][500:-250]
			auxy = im[3,i,:][500:-250]			
		else:
			step = 3501
			auxx = im[0,i,:][200:-200]
			auxy = im[3,i,:][200:-200]

		minim = math.ceil(np.min(auxx))		#aprox al minimo entero
		maxi = math.floor(np.max(auxx))		#aprox al maximo entero
		#plt.plot(im[0,i,:],im[3,i,:],alpha=.4)
		#plt.plot(auxx,auxy)
		#plt.show()
		x = auxx[(auxx < maxi) & (auxx > minim)]	#recorta la con la aprox
		y = auxy[(auxx < maxi) & (auxx > minim)]	
		print 'largos',(len(x)),len(y)
		xdata,step = np.linspace(minim,maxi,(maxi-minim)*linstep+1.,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
		#xdata,step = np.linspace(minim,maxi,(maxi-minim)*8.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.125
	 	#xdata,step = np.linspace(minim,maxi,(maxi-minim)*100.+1,retstep=True)    #genera un x con los extremos de arriba equiespaciado en un 0.001
		
	 	#xdata = np.around(xdata,decimals=1)

	 	if mode == 'spline':
			fit = fit_splines(x,y,xdata,3101)	#Equiespacio los espectros
			#fit = fit_splines(x,y,xdata,901)	#Equiespacio los espectros


		elif mode == 'lineal':	
		 	tck = interpolate.splrep(x,y,k=3)
			fit = interpolate.splev(xdata,tck)
		else:
			print 'Error en opcion mode: elija spline o lineal'
			raise

		xdata = xdata[125:len(xdata)-125]		#Le quito los extremos del fit, que se alejan mucho
		fit = fit[125:len(fit)-125]

		plt.plot(x,y,'bo')
		plt.plot(xdata,fit,'go')

		lineal_x.append(np.array(xdata))
		lineal_y.append(np.array(fit)) 
	plt.show()
	#raise
	return lineal_x,lineal_y

def fit_cleanlines_iter(lineal_x,lineal_y,grado,n):
	waux = []
	faux = []
	ay = []
	fit_pol = []
	auxx = []
	auxy = []
	#plt.plot(lineal_x,lineal_y/cont(lineal_y),'r--')

	for j in np.arange(n):
		if j == 0:
			mask = clean_lines(lineal_x,lineal_y,spectype)
			auxx.append(lineal_x[mask])
			auxy.append(lineal_y[mask])
			#print j
			#print len(auxx[j])
			coefs = poly.polyfit(auxx[j], auxy[j], grado)
			fit = poly.polyval(lineal_x, coefs)
			fit_pol.append(fit)
			ay.append(lineal_y/fit_pol[j])
		else:
			#ay = waux.copy()
			#print j
			#print len(waux[j-1])
			if len (waux[j-1]) == 0:
				continue
			else:
				coefs = poly.polyfit(waux[j-1], faux[j-1], grado)
				fit = poly.polyval(lineal_x, coefs)
				fit_pol.append(fit)				
				ay.append(lineal_y/fit)

		clip = sigma_clip(ay[-1], sigma=2.5)    #aplico sigma clip
		waux1 = np.asarray(lineal_x)[~clip.mask]
		faux1 = np.asarray(lineal_y)[~clip.mask]   
		mask = clean_lines(waux1,faux1,spectype)
		#waux1 = waux1[mask]
		#faux1 = faux1[mask]
		waux.append(waux1[mask])    
		faux.append(faux1[mask])   			

		#if j == 4:
			##para plotiar el ajuste con el sigma clip
			#plt.plot(lineal_x,lineal_y,alpha=.6)
			#plt.plot(lineal_x,fit_pol[j],'r--',alpha=.8)
			#plt.plot(waux[j],faux[j],'k--',alpha=.4)
		#	print ('plot')
	#plt.show()
	#return lineal_y/fit_pol[-1]
	return fit_pol[-1]

#for i in np.arange(len(lineal_x)):
#	fitlineal_y = fit_cleanlines_iter(lineal_x[i],lineal_y[i],1,5)#
#	plt.plot(lineal_x[i],fitlineal_y)
#plt.show()
#raise


def normalizacion(lineal_x,lineal_y):
	auxx = []
	auxy = []
	norm_y = []
	norm_x = []

	#recorro cada orden 
	for i in np.arange(len(lineal_x)):
		fitlineal_y = fit_cleanlines_iter (lineal_x[i],lineal_y[i],1,5)
		
		#plt.plot(lineal_x[i],fitlineal_y,'r')

		#plt.plot(lineal_x[i],lineal_y[i])
		norm_y.append(np.array(lineal_y[i]/fitlineal_y))
		norm_x.append(np.array(lineal_x[i]))				

	#plt.show()

	return norm_x,norm_y


def merge_test(lineal_x,lineal_y):
	#raise
	raw_lineal_x = np.asarray(lineal_x).copy()
	raw_lineal_y = np.asarray(lineal_y).copy()
	wav = []
	flux = []
	auxx = []
	auxy = []
	#Suma coincidencias de ordenes
	for i in np.arange(len(lineal_x)-1):
		mask1 = lineal_x[i+1] >= np.min(lineal_x[i])   #mascara, [i+1] es el orden anterior del [i]. Donde coincide [i+1] con  
		mask2 = lineal_x[i] <= np.max(lineal_x[i+1]) 

		#####tengo que hacer un if .... si el scater es muy muy grande, no lo sumo !!!!

		sumy = (lineal_y[i+1][mask1] + lineal_y[i][mask2])  #guardo la suma de los ordenes
		sumx = lineal_x[i+1][mask1]

		lineal_x[i+1] = lineal_x[i+1][mask1==False]  #recorto el orden anterior
		lineal_y[i+1] = lineal_y[i+1][mask1==False]  

		lineal_x[i] = lineal_x[i][mask2==False]  #recorto el orden
		lineal_y[i] = lineal_y[i][mask2==False]  #recorto el orden


		if len(sumx[clean_lines(sumx,sumy,spectype)]) <= 1:
			#print 'oli'
			continue
		if len(sumy[clean_lines(sumx,sumy,spectype)]) <= 1:
			continue
			#print 'oli'
		if len(sumx) <= 1:
			continue

		lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],2,5)
		lineal_fitysum = fit_cleanlines_iter(sumx,sumy,2,2)
		
		lineal_fity = lineal_y[i]/lineal_fity
		lineal_fitysum = sumy/lineal_fitysum
		##ESTE PLOT
		#plt.plot(sumx,lineal_fitysum,'k')

		#plt.plot(lineal_x[i],lineal_fity)


		wav = np.append(wav,lineal_x[i])
		wav = np.append(wav, sumx)

		flux = np.append(flux, lineal_fity)


		flux = np.append(flux,lineal_fitysum)

	#plt.show()
	return wav,flux

def clean_outliers(wav,flux):
	#cleanflux = clean_lines(wav,flux)
	clip = sigma_clip(flux, sigma=7.0)    #aplico sigma clip
	
	wav = np.asarray(wav)[~clip.mask]

	flux = np.asarray(flux)[~clip.mask]   
	return wav,flux

def gauss(x,a,x0,sigma):
	p = [a, x0, sigma]
	return p[0]* np.exp(-((x-p[1])/p[2])**2)

def ajuste_gauss(x,y):
	mean = sum(x * y) /sum(y)
	sigma = np.sqrt(sum(y * (x - mean)**2.)/ sum(y))
	#dif = (np.max(x) + np.min(x))/2.
	#mask = (x > (dif/10.*2)) & (x < (dif/10.*8))
	print len(y)/10.
	maxi = np.max(y[int(len(y)/10.):-int(len(y)/10.)])
	p0 = [maxi,mean, (x[1]-x[0])*10.]
	fit, tmp = curve_fit(gauss,x,y,p0=p0)
	xnew = np.linspace(x[0],x[-1],len(x))
	gausfit = gauss(x,fit[0],fit[1],fit[2])

	return gausfit

#def limpia_outliers_fitgauss(x,y):
#	gausfit = ajuste_gauss(x,y)


def wavtovel(wav):
	mid = ((np.max(wav)+np.min(wav))/2.)
	return ((wav-mid)/mid)*(c), mid

'''def wavtovel(wav):
	mid = (np.max(wav)-np.min(wav))/2.
	return (wav/mid- 1)*(c*1e13),mid
'''

def veltowav(vel,wav0):
	wav = ((vel/c)*wav0)+wav0
	return wav

def generatevelarray(wav,flux,step):
	mid = ((np.max(wav)+np.min(wav))/2.)
	nvel,mid = wavtovel(wav)
	print nvel[1] - nvel[0]
	plt.plot(nvel,flux)
	plt.show()

	wavi = veltowav(nvel,mid)
	
	plt.plot(wavi,flux,'bo')
	plt.plot(wav,flux,'ro')	
	plt.show()
	
	print (nvel[1] - nvel[0])/10.
	xvel = np.linspace(np.min(nvel),np.max(nvel), len(nvel)*3. )
	tck = interpolate.splrep(nvel,flux,k=3)
	fluxvel = interpolate.splev(xvel,tck)	
	
	plt.plot(xvel,fluxvel,'ro')
	plt.plot(nvel,flux,'bo')
	plt.show()
	raise


	#x = np.linspace(np.min(wav),np.max(wav),len(xvel))

	tck = interpolate.splrep(nvel,flux,k=3)
	fluxvel = interpolate.splev(xvel,tck)

	#plt.plot(wav,flux,'ro')
	#plt.plot(x,yflux,'bo')
	#plt.show()
	#raise
	mid = ((np.max(wav)+np.min(wav))/2.)
	nwav = veltowav(xvel, mid)
	
	#plt.plot(x,yflux,'go')
	#plt.plot(x,yflux,'g')
	#plt.plot(wav,flux,'ro')
	#plt.plot(wav,flux,'r')

	#plt.show()
	#plt.plot(xvel,yflux)
	#plt.show()

	#plt.plot(nwav,yflux,'ko')
	#plt.show()

	return x,yflux,xvel

def rrdeconvolution(w,nf,wav,smod):

	iowav,ioflux,oxvel = generatevelarray(w,nf,-100.,100.,1/300.)
	ismod,isflux,sxvel = generatevelarray(wav,smod,-100.,100.,1/300.)

	tck = interpolate.splrep(iowav,ioflux,k=5)
	ioflux = interpolate.splev(ismod,tck)	
	
	plt.plot(w,nf)
	plt.plot(w,nf,'ro')

	#plt.plot(wav,inf)
	#plt.plot(wav,inf,'bo')

	plt.plot(iowav,ioflux)
	plt.plot(iowav,ioflux,'go')

	plt.plot(ismod,isflux)
	plt.plot(ismod,isflux,'ko')

	plt.show()

	plt.plot(oxvel,ioflux,'r')
	plt.plot(sxvel,isflux,'b')
	plt.show()
	#raise
	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(wav,inf)
	#plt.plot(wav,inf,'bo')

	#plt.plot(wav,smod)
	#plt.plot(wav,smod,'go')

	#plt.show()

	#nn = 1*int(len(ioflux)/(w[-1]-w[0]))
	nn = int(0.1*len(ioflux)/(w[-1]-w[0]))
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	#print len(ismod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	while i < len(ismod)-nn:
		#print i
		vec = ismod[i:i+nn+1]    #smod flux template interpolado al sampleo observado 
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	# and here I reshape the observed spectrum to match with the dimensions of the multiplication of the matrix with the kernel vector
	ww = iowav[int(0.5*nn):-(int(0.5*nn))]
	y = ioflux[int(0.5*nn):-(int(0.5*nn))]

	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel
	#print B
	#print C
	#print D
	x = np.linspace(-len(A)/(w[-1]-w[0])/2.,len(A)/(w[-1]-w[0])/2., len(A))
	
	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	plt.plot(x,A)
	plt.show()
	#plt.plot(A)
	#plt.show()


	return x,A	

def rconvolution(w,nf,wav,smod):

	tck = interpolate.splrep(w,nf,k=3)
	inf = interpolate.splev(wav,tck)
	plt.plot(w,nf)
	plt.plot(w,nf,'ro')

	plt.plot(wav,inf)
	plt.plot(wav,inf,'bo')

	plt.plot(wav,smod)
	plt.plot(wav,smod,'go')

	plt.show()

	nn = 2*int(len(inf)/(w[-1]-w[0]))

	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	#print len(smod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	while i < len(smod)-nn:
		#print i
		vec = smod[i:i+nn+1]    #smod flux template interpolado al sampleo observado 
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	# and here I reshape the observed spectrum to match with the dimensions of the multiplication of the matrix with the kernel vector
	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = inf[int(0.5*nn):-(int(0.5*nn))]

	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel
	#print B
	#print C
	#print D
	x = np.linspace(-len(A)/(w[-1]-w[0])/2.,len(A)/(w[-1]-w[0])/2., len(A))
	
	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	plt.plot(x,A)
	plt.show()
	#plt.plot(A)
	#plt.show()


	return x,A


def veldeconvolution(w,nf,wav,smod):

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	


	#mid = ((np.max(w)+np.min(w))/2.)

	#nvel,_ = wavtovel(w)
	#plt.plot(nvel,nf)
	#plt.show()	
	
	#xvel = np.linspace(np.min(nvel),np.max(nvel), len(nvel) )
	#tck = interpolate.splrep(nvel,nf,k=3)
	#fluxvel = interpolate.splev(xvel,tck)	

	#xvel = np.linspace(np.min(mid),np.max(mid), len(mid) )
	#tck = interpolate.splrep(mid,nf,k=3)
	#fluxvel = interpolate.splev(nmid,tck)	

	nn = 2*int(len(nf)/(w[-1]-w[0]))


	#nn = 2*int(len(fluxvel)/(xvel[-1]-xvel[0]))	
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	#print len(nmod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	midwav = []

	while i < len(nmod)-nn:
		#print i
		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		synvel,mid = wavtovel(w[i:i+nn+1])
		midwav = np.append(midwav,mid)
		vecflux = nmod[i:i+nn+1]
		linsynvel = np.linspace(np.min(synvel),np.max(synvel), len(synvel) )
		#plt.plot(synvel,vecflux)
		#plt.show()
		tck = interpolate.splrep(synvel,vecflux,k=3)
		linsynfluxvel = interpolate.splev(linsynvel,tck)					
		
		vec = linsynfluxvel.copy()
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat
	xvel = np.linspace(np.min(midwav),np.max(midwav), len(midwav) )
	tck = interpolate.splrep(w,nf,k=3)
	fluxvel = interpolate.splev(xvel,tck)

	#plt.plot(w,nf,'ro')
	#plt.plot(xvel,fluxvel,'bo')
	#plt.show()

	# we dont need reshape
	ww = xvel.copy()
	y = fluxvel.copy()	
	####longitud de onda de sampleo del sintetico ..... el y y el ww deberia estar en longitud de onda que ocupo para llevar a velocidades (wav central)
	## hacer interpolacion lineal ... en vez de splines 
	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel
	#print B
	#print C
	#print D
	xvel,_ = wavtovel(xvel)

	x = np.linspace(-len(A)/(xvel[-1]-xvel[0])/2.,len(A)/(xvel[-1]-xvel[0])/2., len(A))
	
	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	#plt.plot(x,A)
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	
	#plt.show()
	plt.plot(x,A)
	#plt.show()

	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(w,nmod)
	#plt.plot(w,nmod,'go')

	#plt.show()
	return x,A

def rconvolution(w,nf,wav,smod):

	tck = interpolate.splrep(w,nf,k=3)
	inf = interpolate.splev(wav,tck)
	plt.plot(w,nf)
	plt.plot(w,nf,'ro')

	#plt.plot(wav,inf)
	#plt.plot(wav,inf,'bo')

	#plt.plot(wav,smod)
	#plt.plot(wav,smod,'go')

	#plt.show()

	nn = 2*int(len(inf)/(w[-1]-w[0]))

	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	print len(smod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	while i < len(smod)-nn:
		#print i
		vec = smod[i:i+nn+1]    #smod flux template interpolado al sampleo observado 
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	# and here I reshape the observed spectrum to match with the dimensions of the multiplication of the matrix with the kernel vector
	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = inf[int(0.5*nn):-(int(0.5*nn))]

	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel
	#print B
	#print C
	#print D
	x = np.linspace(-len(A)/(w[-1]-w[0])/2.,len(A)/(w[-1]-w[0])/2., len(A))
	
	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	plt.plot(x,A)
	#plt.show()
	#plt.plot(A)
	#plt.show()


	return x,A


def rafa_deconvolution(w,nf,wav,smod):
	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	#print 'w,nf,wav,smod',len(w),len(nf),len(wav),len(smod)
	#print 'w,nf,wav,smod',w[1]-w[0],w[1]-w[0],w[1]-w[0],smod[1]-smod[0]
	#plt.plot(w,nf)	
	#plt.show()
	# normalization, just in case...
	#nf /= np.add.reduce(nf)
	#nmod /= np.add.reduce(nmod)
	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(w,nmod)
	#plt.plot(w,nmod,'go')

	#plt.show()
	#raise
	#plot(w,nmod)
	#3plot(wav,smod/np.add.reduce(smod))
	#show()
	#I will consider that the kernel has a width of 2 amstrongs, and here I compute how many pixels do I need for that.
	
	nn = 2*int(len(nf)/(w[-1]-w[0]))

	# Now I generate the matrix of the template spectrum
	mat = []
	i=0
	print len(nmod)-nn 			#len(nmod)-nn es el largo que tendra el nuevo template
	#raise
	while i < len(nmod)-nn:
		#print i
		vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	# and here I reshape the observed spectrum to match with the dimensions of the multiplication of the matrix with the kernel vector
	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = nf[int(0.5*nn):-(int(0.5*nn))]
	print 'nf y w',len(nf), len(w)

	print 'len y y len mat', len(y), len(mat)

	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
	#print A # this is the kernel
	#print B
	#print C
	#print D
	#x = np.linspace(-len(A)/(w[-1]-w[0])/2.,len(A)/(w[-1]-w[0])/2., len(A))
	x = np.linspace(-len(A)/2.,len(A)/2.,len(A)  )
	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	#plt.plot(x,A)
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	
	#plt.show()
	#plt.plot(A)
	#plt.show()

	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(w,nmod)
	#plt.plot(w,nmod,'go')

	#plt.show()
	return x,A

def condicion_linea(wav):
	a = False
	if (np.min(wav_aux) < 6565.) & (np.max(wav_aux) > 6565):
		a = True
	if (np.min(wav_aux) < 4863.) & (np.max(wav_aux) > 4863):
		a = True
	if (np.min(wav_aux) < 4342.) & (np.max(wav_aux) > 4342):
		a = True
	if (np.min(wav_aux) < 5180.) & (np.max(wav_aux) > 5180):
		a = True
	if (np.min(wav_aux) < 5380.) & (np.max(wav_aux) > 5380):
		a = True
	if (np.min(wav_aux) < 5650.) & (np.max(wav_aux) > 5650):
		a = True
	if (np.min(wav_aux) < 5850.) & (np.max(wav_aux) > 5850):
		a = True
	if (np.min(wav_aux) < 6080.) & (np.max(wav_aux) > 6080):
		a = True
	if (np.min(wav_aux) < 6480.) & (np.max(wav_aux) > 6480):
		a = True

	return a 

parser = argparse.ArgumentParser(description='Linealizacion Merge y deconvolucion')
parser.add_argument('spec', help='directorio/archivo')
parser.add_argument('--vel', type=float, default=-17.7710, help='velocidad orbita')
parser.add_argument('--rango', type=float, default=100.0, help='rango deconvolucionado')
parser.add_argument('--template', type=str, default='ap00t6250g50k0odfnew_sample0.005.out', help='nombre template')
#parser.add_argument('--bkg', type=str, default=None, help='Para estimar bkg, ingresar Yes')
parser.add_argument('--linstep', type=float, default=50, help='step para linealizacion')
parser.add_argument('--mode', type=str, default='spline', help='modo de linealizacion spline y lineal')
parser.add_argument('--spectype', type=str, default='F', help='Tipo espectral F o A')
parser.add_argument('--deconv', type=str, default='vel', help='tipo de deconvolucion, en wav o vel')

args = parser.parse_args()
spec  = args.spec
vel = args.vel
rango = args.rango
template = args.template
linstep = args.linstep
mode = args.mode
spectype = args.spectype
deconv = args.deconv

#leo espectros, los corrijo por velocidad
hdulist = pyfits.open(spec)
hdulist.info()
im = pyfits.getdata(spec)

im[0,:,:] = shift_doppler(im[0,:,:],vel)
x = im [0,:,:]
y = im [1,:,:]

#leo template
if template != 'ap00t6250g50k0odfnew_sample0.005.out':
	hmod = pyfits.getheader('6500_40_p00p00.ms.fits')
	smod = pyfits.getdata('6500_40_p00p00.ms.fits')[0]
	flux_syn = pyfits.getdata('6500_40_p00p00.ms.fits')[0]
	wav_syn = np.arange(len(smod))*hmod['CDELT1']+hmod['CRVAL1']
	wav_syn = ToVacuum(wav_syn)

elif template == 'ap00t6250g50k0odfnew_sample0.005.out':

	data = np.loadtxt(template,usecols=[0,1],unpack=True)
	wav_syn = ToVacuum(data[0])
	flux_syn = data[1]

	
for i in np.arange(len(im[0,:,0])):#
	#im[0,i,:] = im[0,i,:][200:-200]#
	#im[3,i,:] = im[3,i,:][200:-200]
	xa = im [0,i,:]
	ya = im [3,i,:]
	plt.plot(xa,ya)
plt.show()
#raise

#linealizo, puede ser con spline o con interpolacion lineal
lineal_x, lineal_y = linealizacion(im,linstep,mode)

#PLOT
#for i in np.arange(len(lineal_x)):
	#x = lineal_x[i][10:]
	#y = lineal_y[i][10:]
	#plt.plot(x,y,'o')
#plt.show()

#Saco las lineas anchas. Normalizo cada orden
#norm_x, norm_y = normalizacion(lineal_x,lineal_y)

#PLOT
#for i in np.arange(len(norm_x)):
#	x = norm_x[i]
#	y = norm_y[i]
#	plt.plot(x,y,'o')
#plt.show()
#print len(max(lineal_x,key=len))


# MERGE: Saco las lineas anchas. Normalizo cada orden con spline o cuadratica.
# Sumo coicidencia entre ordenes. corrijo trend con cuadratica. 
wav,flux = merge_test(lineal_x,lineal_y)
#ordeno y limpio outliers
wav,flux = ordenar(wav,flux)
wav,flux = clean_outliers(wav,flux)

#plt.plot(wav,flux)
#plt.plot(wav_syn1,flux_syn1)
#plt.show()

#guardo
#table = addcolsbyrow(wav,flux)
#table.write("WASP24_test.txt", format='ascii')


#rango = 100.
num = (np.max(wav)-np.min(wav))/rango
num = np.floor(num)
print num
x = []
y = []
plt.plot(wav,flux)
plt.show()

#lines = [6565,4863,4342] #5180 5380 5650 5850 6080 6480
for i in np.arange(num-1):
	i = i +1.
	print 'rango:',np.min(wav)+rango*(i),'to', np.min(wav)+rango*(1+i)
	#selecciono rango de 100 A
	mask = (wav > np.min(wav)+rango*(i) ) & (wav < np.min(wav)+rango*(1+i) )
	mask_syn = (wav_syn > np.min(wav)+rango*(i) ) & (wav_syn < np.min(wav)+rango*(1+i) )
	wav_aux = wav[mask]
	flux_aux = flux[mask]
	flux_syn_aux = flux_syn[mask_syn]
	wav_syn_aux = wav_syn[mask_syn]
	
	#reviso si hay lineas gruesas en el rango seleccionado, si hay, lo salto
	if condicion_linea(wav_aux,) == True:
		continue
	else:
		print 'en este rango no hay lineas gruesas'
	#plt.plot(wav_aux,flux_aux)
	#plt.plot(wav_aux,flux_aux,'ro')
	#plt.plot(wav_syn_aux,flux_syn_aux)
	#plt.plot(wav_syn_aux,flux_syn_aux,'bo')

	#corrijo con ajuste lineal, itero 5 para encontrar el mejor fit
	fitspec = fit_cleanlines_iter(wav_aux,flux_aux,1,5)
	fitsyn = fit_cleanlines_iter(wav_syn_aux,flux_syn_aux,1,5)
	#plt.plot(wav_aux,fitspec,'r')
	#plt.plot(wav_syn_aux,fitsyn,'g')
	#plt.show()
	dif = np.median(fitspec)-np.median(fitsyn)
	flux_aux = flux_aux-dif 

	#elimino lineas anchas
	mask = clean_lines(wav_aux,flux_aux,spectype)
	wav_aux = wav_aux[mask]
	flux_aux = flux_aux[mask] 
	mask = clean_lines(wav_syn_aux,flux_syn_aux,spectype)
	flux_syn_aux = flux_syn_aux[mask] 
	wav_syn_aux = wav_syn_aux[mask]

	#xi,yi = rconvolution(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
	if deconv == 'vel':
		xi,yi = veldeconvolution(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
	elif deconv == 'wav':
		xi,yi = rafa_deconvolution(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
	else:
		print 'error --decov: solo puede ser vel o wav'
		raise

	#elimino deconvoluciones muy outliers

	gausfit = ajuste_gauss(xi,yi)

	lim = int(len(yi)/10.*3)
	short = np.std(yi[lim:-lim]/gausfit[lim:-lim])
	print 'std',short
	plt.plot(xi,gausfit,'ro')
	plt.plot(xi[lim:-lim],yi[lim:-lim]/gausfit[lim:-lim])
	#plt.plot(xi[mask],yi[mask])
	plt.show()
	if short > 0.2:
		continue
	else:
		x.append(xi)
		y.append(yi)
		#plt.plot(xi,yi)
		#plt.show()

	#plot muestra mean de deconvolucion con i=8 rangos
	'''if i == 8:
		aux = []
		print ('empieza contar largos')
		for i in np.arange(len(y)):
			aux = np.append(aux,len(y[i]))
		print ('empieza maximo')

		for j in np.arange(len(y)):
			print int(np.min(aux))
			y[j] = y[j][0:int(np.min(aux))]
		ymean = np.mean(y,axis=0)
		xmean = x[0].copy()
		print ('empieza condiciones')

		if (len(x[0]) > len(ymean)):
			xmean = xmean[0:len(ymean)]
		elif (len(x[0]) < len(ymean)):
			ymean = ymean[0:len(x[0])]

		print('plot')
		plt.plot(xmean,ymean)
		plt.show()
	'''
	#print 'termino'

print "termino deconvolucion"
for i in np.arange(len(x)):
	plt.plot(x[i],y[i])
plt.show()

##las deconvoluciones tienen distinto largo (dif de 2 o 3 elementos)
##los dejo de la misma dimension
aux = []
for i in np.arange(len(y)):
	aux = np.append(aux,len(y[i]))
for j in np.arange(len(y)):
	y[j] = y[j][0:int(np.min(aux))]
	x[j] = x[j][0:int(np.min(aux))]

ymean = np.mean(y,axis=0)
xmean = x[0].copy()

print "plot mean"
plt.plot(xmean,ymean,'bo')
plt.show()


'''
xnew = []
ynew = []
for i in np.arange(len(x)):
	print len(x), len(x[i])	
	print len(y), len(y[i])
	print len(x[0])
	#plt.plot(x[i],y[i])
	#plt.show()
	xlin = np.linspace(np.min(x[0]),np.max(x[0]),len(x[0])*10)
	tck = interpolate.splrep(x[i],y[i],k=3)
	ylin = interpolate.splev(xlin,tck)
	xnew.append(xlin)
	ynew.append(ylin)
	plt.plot(xlin,ylin)
plt.show()

ynewmean = np.mean(ynew,axis=0)
xnewmean = xnew[0].copy()
print ('empieza condiciones')

if (len(xnew[0]) > len(ynewmean)):
	xnewmean = xnewmean[0:len(ynewmean)]
elif (len(x[0]) < len(ymean)):
	ynewmean = ynewmean[0:len(x[0])]

#ymean = np.mean(y,axis=0)

plt.plot(xnewmean,ynewmean)
plt.show()

print ('empieza condiciones')
if (len(x[0]) > len(ymean)):
	print 'se cumple 1'
	xmean = xmean[0:len(ymean)]
elif (len(x[0]) < len(ymean)):
	print 'se cumple 1'

	ymean = ymean[0:len(x[0])]
else:
	'no se cumple nada
#ymean = np.mean(y,axis=0)


'''




