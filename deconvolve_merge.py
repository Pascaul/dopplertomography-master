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
import collections as collections
from PyAstronomy import funcFit as fuf
from specutils.io import read_fits
from astropy.convolution import Trapezoid1DKernel


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
		mask4 = ((wav > 4855) & (wav < 4878))
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
		#im[0,i,:] = shift_doppler(im[0,i,:],vels)
		print 'max', np.max(im[0,i,:])
		print len (im[0,i,:])
		wav = im[0,i,:]
		flux = im[3,i,:] 

		if np.max(im[0,i,:]) <= 5500:
			print 'esta condicion'
			mask = (wav < np.max(wav)-4) & (wav > np.min(wav) + 5) 
			step = 3201
			auxx = wav[mask]
			auxy = flux[mask]			
			#auxx = wav[150:-150]
			#auxy = flux[150:-150]
			#auxx = im[0,i,:][500:-250]
			#auxy = im[3,i,:][500:-250]			
		else:
			step = 3501
			mask = (wav < np.max(wav) -1) & (wav > np.min(wav) + 1) 
			auxx = wav[mask]
			auxy = flux[mask]
			#auxx = im[0,i,:][200:-200]
			#auxy = im[3,i,:][200:-200]

		minim = math.ceil(np.min(auxx))		#aprox al minimo entero
		maxi = math.floor(np.max(auxx))		#aprox al maximo entero
		#plt.plot(im[0,i,:],im[3,i,:],alpha=.4)
		#plt.plot(auxx,auxy)
		#plt.show()
		x = auxx[(auxx < maxi) & (auxx > minim)]	#recorta la con la aprox
		y = auxy[(auxx < maxi) & (auxx > minim)]	
		print 'largos',(len(x)),len(y)
		if len(x) == 0:
			continue

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

		mask = (xdata < np.max(xdata) -3) & (xdata> np.min(xdata) + 3) 
			
		xdata = xdata[mask]		#Le quito los extremos del fit, que se alejan mucho
		fit = fit[mask]
		#xdata = xdata[125:len(xdata)-125]		#Le quito los extremos del fit, que se alejan mucho
		#fit = fit[125:len(fit)-125]

		plt.plot(x,y,'bo')
		plt.plot(xdata,fit,'go')

		lineal_x.append(np.array(xdata))
		lineal_y.append(np.array(fit)) 
	plt.show()
	#raise
	return lineal_x,lineal_y



def linealizacion_tres(spectra_list,linstep,mode):
	lineal_x = []
	lineal_y = []

	for i in np.arange(len(spectra_list)):
		wav = spectra_list[i].wavelength
		flux = spectra_list[i].flux	
		print wav
		wav = shift_doppler(wav,vel)
		print wav
		print len(wav)
		wav = np.asarray(wav)
		flux = np.asarray(flux)

		#print 'max', np.max(im[0,i,:])
		#print len (im[0,i,:])
		
		if np.max(wav) >= 6500:
			continue
		if np.max(wav) <= 4700:
			print 'esta condicion en', np.max(wav)
			mask = (wav < np.max(wav)-8) & (wav > np.min(wav) + 10) 
			step = 3201
			#auxx = wav[450:-300]
			#auxy = flux[450:-300]
			auxx = wav[mask]
			auxy = flux[mask]			
		else:
			step = 3501
			mask = (wav < np.max(wav) -6) & (wav > np.min(wav) + 5) 

			auxx = wav[mask]
			auxy = flux[mask]
			#auxx = wav[150:-150]
			#auxy = flux[150:-150]
		
		if np.max(wav) >= 6500:
			continue
		if np.max(wav) <= 3900:
			continue
		minim = math.ceil(np.min(auxx))		#aprox al minimo entero
		maxi = math.floor(np.max(auxx))		#aprox al maximo entero
		#plt.plot(im[0,i,:],im[3,i,:],alpha=.4)
		#plt.plot(auxx,auxy)
		#plt.show()
		x = auxx[(auxx < maxi) & (auxx > minim)]	#recorta la con la aprox
		y = auxy[(auxx < maxi) & (auxx > minim)]	
		print 'largos',(len(x)),len(y)
		if len(x) == 0:
			continue

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

		#xdata = xdata[125:len(xdata)-125]		#Le quito los extremos del fit, que se alejan mucho
		#fit = fit[125:len(fit)-125]
		mask = (xdata < np.max(xdata)-4) & (xdata > np.min(xdata) + 4) 
		xdata = xdata[mask]
		fit = fit[mask]

		plt.plot(x,y,'bo')
		plt.plot(xdata,fit,'go')

		lineal_x.append(np.asarray(xdata))
		lineal_y.append(np.asarray(fit)) 
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
	#m = 0
	#mask = clean_lines(lineal_x,lineal_y,spectype)
	####tengo errores, si hay erroroes solo ajustar lineal en linealx,linealy, sin borrar lineas ... 
	#if len(lineal_y[mask]) == 0:
	#	print 'error'
	#	return np.ones(len(lineal_x))

	for j in np.arange(n):
		if j == 0:
			mask = clean_lines(lineal_x,lineal_y,spectype)
			auxx.append(lineal_x[mask])
			auxy.append(lineal_y[mask])
			#print j
			#print len(auxx[j])
			#if len(auxx[0]) == 0:

			#	continue
			coefs = poly.polyfit(auxx[j], auxy[j], grado)
			fit = poly.polyval(lineal_x, coefs)
			fit_pol.append(fit)
			ay.append(lineal_y/fit_pol[j])
		else:
			#ay = waux.copy()
			#print j
			#if len(auxx[0]) == 0 :
		#		continue
			#print 'largo',len(waux[j-1])
			#if (len (waux[j-1]) == 0):
			#	ay.append(np.ones(len(lineal_y)))
			#	continue
			#else:
			coefs = poly.polyfit(waux[j-1], faux[j-1], grado)
			fit = poly.polyval(lineal_x, coefs)
			fit_pol.append(fit)				
			ay.append(lineal_y/fit)

		clip = sigma_clip(ay[-1], sigma=2.0)    #aplico sigma clip
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
	#plt.plot(lineal_x,lineal_y,alpha=.4)
	#plt.plot(lineal_x, fit_pol[-1] )
	plt.plot(waux[-1],faux[-1])
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

def merge(lineal_x,lineal_y):
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

		##si no hay coincidencias
		if np.count_nonzero(mask1) == len(lineal_y[i+1]):
			lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],2,5)
			plt.plot(lineal_x[i],lineal_y[i],alpha=.3)
			plt.plot(lineal_x[i],lineal_fity)		

			lineal_fity = lineal_y[i]/lineal_fity
			wav = np.append(wav,lineal_x[i])
			flux = np.append(flux, lineal_fity)

		else:

			sumy = (lineal_y[i+1][mask1] + lineal_y[i][mask2])  #guardo la suma de los ordenes
			sumx = lineal_x[i+1][mask1]

			lineal_x[i+1] = lineal_x[i+1][mask1==False]  #recorto el orden anterior
			lineal_y[i+1] = lineal_y[i+1][mask1==False]  

			lineal_x[i] = lineal_x[i][mask2==False]  #recorto el orden
			lineal_y[i] = lineal_y[i][mask2==False]  #recorto el orden


			
			lineal_fity = fit_cleanlines_iter(lineal_x[i],lineal_y[i],2,5)
			plt.plot(lineal_x[i],lineal_fity)

			lineal_fity = lineal_y[i]/lineal_fity
			wav = np.append(wav,lineal_x[i])
			flux = np.append(flux, lineal_fity)		

			if len(sumx[clean_lines(sumx,sumy,spectype)]) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
			elif len(sumy[clean_lines(sumx,sumy,spectype)]) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
				#print 'oli'
			elif len(sumx) <= 1:
				print 'el orden', i, 'no se solapa con el',i+1
			else:
				lineal_fitysum = fit_cleanlines_iter(sumx,sumy,2,2)		
				plt.plot(sumx,sumy,alpha=.3)

				plt.plot(sumx,lineal_fitysum,'k')

				lineal_fitysum = sumy/lineal_fitysum
				wav = np.append(wav, sumx)
				flux = np.append(flux,lineal_fitysum)



						##ESTE PLOT

	plt.show()
	return wav,flux

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
		#print 'mascaras',len(lineal_y[i+1]),len(mask1)
		#print 'aplicar mascara', len(lineal_y[i+1][mask1])
		#print 'mascaras',len(lineal_y[i]),len(mask2)
		#print 'aplicar mascara', len(lineal_y[i][mask2])

		if len(lineal_y[i+1][mask1]) == len (lineal_y[i][mask2]):  ###aplicar solo si los ordenes se solapan

			sumy = (lineal_y[i+1][mask1] + lineal_y[i][mask2])  #guardo la suma de los ordenes
			sumx = lineal_x[i+1][mask1]

			lineal_x[i+1] = lineal_x[i+1][mask1==False]  #recorto el orden anterior
			lineal_y[i+1] = lineal_y[i+1][mask1==False]  

			lineal_x[i] = lineal_x[i][mask2==False]  #recorto el orden
			lineal_y[i] = lineal_y[i][mask2==False]  #recorto el orden


		else:
			print 'el orden', i, 'no se solapa con el',i+1

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
		plt.plot(sumx,lineal_fitysum,'k')
		plt.plot(lineal_x[i],lineal_fity)
		wav = np.append(wav,lineal_x[i])
		wav = np.append(wav, sumx)
		flux = np.append(flux, lineal_fity)
		flux = np.append(flux,lineal_fitysum)

	plt.show()
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


def wavstovel(wav):
	mid = ((np.max(wav)+np.min(wav))/2.)
	return ((wav-mid)/mid)*(c), mid


def resamplewav(wav):
	mid = (np.max(wav)-np.min(wav))/2.
	vels = np.linspace(-100.,100.,2e2)

	return  mid / (1 - (vels/c))

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


def veldeconvolution2(w,nf,wav,smod):

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	nn = 10*int(len(nf)/(w[-1]-w[0]))
	
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0

	midwav = []
	stepvel = []
	print 'step',len(w[i:i+nn+1])
	j = 0
	vels = np.linspace(-100,100,300)
	while i < len(nmod)-nn:

		#print i
		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		#genero arreglo de vel a partir de las wav, guardo wav centrales
		#synwav_samplevel = resamplewav(w[i:i+nn+1])
		mid = np.median(w[i:i+nn+1])
		#mid = (np.max(w[i:i+nn+1]) + np.min(w[i:i+nn+1]))/2.
		
		new_wav = mid / (1. - (vels/c))
		#genero un vector de la matriz de deconvolucion
		#linealizo
		tck = interpolate.splrep(w[i:i+nn+1],nmod[i:i+nn+1],k=1)
		new_flux = interpolate.splev(new_wav,tck)
		#plt.plot(w[i:i+nn+1],nmod[i:i+nn+1],'ro')
		#plt.plot(new_wav,new_flux,'bo')
		#plt.show()
		#print 'min y max', np.min(w[i:i+nn+1]), np.max(w[i:i+nn+1])
		#print 'min y max new', np.min(new_wav),np.max(new_wav)
		#print 'mid',mid
		#print new_wav
		#print new_flux
		#print 'nan',np.isnan(nmod[i:i+nn+1]).sum()
		if 	np.isnan(new_flux).sum() != 0 :

			print np.isnan(new_flux).sum()
			print np.isnan(new_wav).sum()
			#plt.plot(new_wav,new_flux,'bo')
			#plt.show()
			j +=1
			print ('se corta') 
			new_flux = np.ones(len(new_flux))

			#return np.asarray([0]),np.asarray([0])

		#new_flux, new_wav = ordenar(new_wav,new_flux)				
		#plt.plot(w[i:i+nn+1],nmod[i:i+nn+1],'ro')
		#plt.plot(new_wav,new_flux,'bo')
		#plt.show()
		vec = new_flux.copy()
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = nf[int(0.5*nn):-(int(0.5*nn))]

	#plt.plot(w,nf,'ro')
	#plt.plot(xvel,fluxvel,'bo')
	#plt.show()
	#raise
	# we dont need reshape

	####longitud de onda de sampleo del sintetico ..... el y y el ww deberia estar en longitud de onda que ocupo para llevar a velocidades (wav central)
	## hacer interpolacion lineal ... en vez de splines 
	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
		#print A # this is the kernel
	#print B
	#print C
	#print D

	#x = np.linspace(-len(A)/(xvel[-1]-xvel[0])/2.,len(A)/(xvel[-1]-xvel[0])/2., len(A))
	#x = np.linspace(-len(A)/np.min(stepvel)/2.,len(A)/np.min(stepvel)/2., len(A))

	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	#plt.plot(x,A)
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')

	#raise
	#plt.show()
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	#plt.plot(vels,A)
	#plt.show()

	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(w,nmod)
	#plt.plot(w,nmod,'go')
	#plt.plot(vels,A)
	#plt.show()
	#raise
	return np.array(vels),np.array(A)


def veldeconvolution3(w,nf,wav,smod):

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	nn = 6*int(len(nf)/(w[-1]-w[0]))
	
	# Now I generate the matrix of the template spectrum
	mat = []
	i=0

	midwav = []
	stepvel = []
	print 'step',len(w[i:i+nn+1])

	#velsfix = np.linspace(-100,100,200)
	velsfix = np.linspace(-80,80,200)

	auxwav = []
	while i < len(nmod)-nn:
		#print i
		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		#genero arreglo de vel a partir de las wav, guardo wav centrales
		#synwav_samplevel = resamplewav(w[i:i+nn+1])
		mid = np.median(w[i:i+nn+1])
		#mid = (np.max(w[i:i+nn+1]) + np.min(w[i:i+nn+1]))/2.
		
		vels,midwav = wavstovel(w[i:i+nn+1])
		auxwav.append(midwav)
		#genero un vector de la matriz de deconvolucion
		#linealizo
		tck = interpolate.splrep(vels,nmod[i:i+nn+1],k=3)
		new_flux = interpolate.splev(velsfix,tck)
		if 	np.isnan(new_flux).sum() != 0:

			#print np.isnan(new_flux).sum()
			#print np.isnan(new_wav).sum()
			#plt.plot(new_wav,new_flux,'bo')
			#plt.show()
			return np.asarray([0]),np.asarray([0])

		#new_flux, new_wav = ordenar(new_wav,new_flux)				
		#plt.plot(w[i:i+nn+1],nmod[i:i+nn+1],'ro')
		#plt.plot(new_wav,new_flux,'bo')
		#plt.show()

		vec = new_flux.copy()
		if len(mat) == 0:
			mat = vec
		else:
			mat = np.vstack((mat,vec))
		i+=1
		#print mat

	ww = w[int(0.5*nn):-(int(0.5*nn))]
	y = nf[int(0.5*nn):-(int(0.5*nn))]

	#midwav = np.linspace(np.min(auxwav),np.max(auxwav), len(auxwav) )
	#tck = interpolate.splrep(ww,y,k=3)
	#midflux = interpolate.splev(midwav,tck)


	#plt.plot(w,nf,'ro')
	#plt.plot(xvel,fluxvel,'bo')
	#plt.show()
	#raise
	# we dont need reshape

	####longitud de onda de sampleo del sintetico ..... el y y el ww deberia estar en longitud de onda que ocupo para llevar a velocidades (wav central)
	## hacer interpolacion lineal ... en vez de splines 
	# lstsq solution of the keting the kerner
	A,B,C,D = np.linalg.lstsq(mat, y, rcond=None)
		#print A # this is the kernel
	#print B
	#print C
	#print D

	#x = np.linspace(-len(A)/(xvel[-1]-xvel[0])/2.,len(A)/(xvel[-1]-xvel[0])/2., len(A))
	#x = np.linspace(-len(A)/np.min(stepvel)/2.,len(A)/np.min(stepvel)/2., len(A))

	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	#plt.plot(x,A)
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')

	#raise
	#plt.show()
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	#plt.plot(vels,A)
	#plt.show()

	#plt.plot(w,nf)
	#plt.plot(w,nf,'ro')

	#plt.plot(w,nmod)
	#plt.plot(w,nmod,'go')

	#plt.show()
	return np.array(velsfix),np.array(A)



def veldeconvolution(w,nf,wav,smod):

	tck = interpolate.splrep(wav,smod,k=3)
	nmod = interpolate.splev(w,tck)
	
	####esta todo sampleado a w

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
	stepvel = []
	while i < len(nmod)-nn:
		#print i
		#vec = nmod[i:i+nn+1]    #nmod flux template interpolado al sampleo observado 
		#genero arreglo de vel a partir de las wav, guardo wav centrales
		synvel,mid = wavstovel(w[i:i+nn+1])
		midwav = np.append(midwav,mid)
		#genero un vector de la matriz de deconvolucion
		vecflux = nmod[i:i+nn+1]
		#linealizo
		linsynvel = np.linspace(np.min(synvel),np.max(synvel), len(synvel) )
		stepvel = np.append(stepvel,linsynvel[1]-linsynvel[0])
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
	print np.min(stepvel),np.max(stepvel)
	#raise
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
	xvel,_ = wavstovel(xvel)

	#x = np.linspace(-len(A)/(xvel[-1]-xvel[0])/2.,len(A)/(xvel[-1]-xvel[0])/2., len(A))
	x = np.linspace(-len(A)/np.min(stepvel)/2.,len(A)/np.min(stepvel)/2., len(A))

	#print ('std',np.std(A))
	#print ('std short',np.std(A[(x < 0.2) & (x > -0.2)]))
	#plt.plot(x,A)
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	bini = xvel[1]-xvel[0]
	dif = (xvel[-1]+xvel[0])/2.
	print xvel
	print bini, dif
	#raise
	#plt.show()
	#plt.title('obs-spec step %1.3f ,obs-spec lineal step %1.3f, syn-spec step %1.3f, sync-spec lineal step %1.3f' %( b,  (w[1]-w[0]), (wav[1]- wav[0]), (w[1]- w[0]) ))
	#plt.xlabel('wav A')
	#plt.plot(x,A)
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


def wav_deconvolution(w,nf,wav,smod):
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
	
	nn = 6*int(len(nf)/(w[-1]-w[0]))

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
	plt.xlabel('wav A')
	
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

def vsin_kernel(wave_spec,flux_spec, vrot, epsilon):
	#wave_ = np.log(wave_spec) 
	#velo_ = np.linspace(wave_[0],wave_[-1],len(wave_)) 
	#flux_ = np.interp(velo_,wave_,flux_spec) 
	wave_ = np.log(wave_spec) 
	#wave_ = wave_spec.copy()
	velo_ = np.linspace(wave_[0],wave_[-1],len(wave_)) 
	flux_ = flux_spec.copy()
	#plt.plot(wave_spec,flux_spec,'bo')
	#plt.plot(wave_spec,flux_,'ro')
	#plt.show()
	dvelo = velo_[1]-velo_[0] 
	#vrot = vrot/(c*1e-3) 
	vrot_ = vrot/(c) 

	#-- compute the convolution kernel and normalise it 
	n = int(2*vrot_/dvelo) 
	velo_k = np.arange(n)*dvelo 
	velo_k -= velo_k[-1]/2. 
	y = 1 - (velo_k/vrot_)**2 # transformation of velocity 
	G = (2*(1-epsilon)*np.sqrt(y)+np.pi*epsilon/2.*y)/(np.pi*vrot_*(1-epsilon/3.0))  # the kernel 
	G /= G.sum() 
	#-- convolve the flux with the kernel 

	flux_conv = np.convolve(1-flux_,G,mode='same') 
	velo_ = np.arange(len(flux_conv))*dvelo+velo_[0] 
	wave_conv = np.exp(velo_) 

	#vrot = vrot
	#-- compute the convolution kernel and normalise it 
	#n = int(2*vrot/dvelo) 
	#velo_k = np.arange(n)*dvelo 
	#velo_k -= velo_k[-1]/2. 

	return G,velo_k

def broadGaussFast(x, y, sigma, edgeHandling=None, maxsig=None):
    """
    Apply Gaussian broadening. 
    This function broadens the given data using a Gaussian
    kernel.
    Parameters
    ----------
    x, y : arrays
        The abscissa and ordinate of the data.
    sigma : float
        The width (i.e., standard deviation) of the Gaussian
        profile used in the convolution.
    edgeHandling : string, {None, "firstlast"}, optional
        Determines the way edges will be handled. If None,
        nothing will be done about it. If set to "firstlast",
        the spectrum will be extended by using the first and
        last value at the start or end. Note that this is
        not necessarily appropriate. The default is None.
    maxsig : float, optional
        The extent of the broadening kernel in terms of
        standard deviations. By default, the Gaussian broadening
        kernel will be extended over the entire given spectrum,
        which can cause slow evaluation in the case of large spectra.
        A reasonable choice could, e.g., be five.  
    Returns
    -------
    Broadened data : array
        The input data convolved with the Gaussian
        kernel.
    """
    # Check whether x-axis is linear
    dxs = (x[1:] - x[0:-1])*0.1

    #if abs(max(dxs) - min(dxs)) > np.mean(dxs) * 1e-6:
    #	print ('wavs no son equidistantes!')
    #    raise
    if maxsig is None:
        lx = len(x)
    else:
        lx = int(((sigma * maxsig) / dxs[0]) * 2.0) + 1
    # To preserve the position of spectral lines, the broadening function
    # must be centered at N//2 - (1-N%2) = N//2 + N%2 - 1
    nx = (np.arange(lx, dtype=np.int) - sum(divmod(lx, 2)) + 1) * dxs[0]
    gf = fuf.GaussFit1d()
    gf["A"] = 1.0
    gf["sig"] = sigma
    e = gf.evaluate(nx)
    # This step ensured that the
    e /= np.sum(e)

    '''if edgeHandling == "firstlast":
                    nf = len(y)
                    y = np.concatenate((np.ones(nf) * y[0], y, np.ones(nf) * y[-1]))
                    result = np.convolve(y, e, mode="same")[nf:-nf]
                elif edgeHandling is None:
                    result = np.convolve(y, e, mode="same")
                else:
                    raise(PE.PyAValError("Invalid value for `edgeHandling`: " + str(edgeHandling),
                                         where="broadGaussFast",
                                         solution="Choose either 'firstlast' or None"))'''
    return e,nx



def instrBroadGaussFast(wvl, flux, resolution, edgeHandling=None, fullout=False, maxsig=None):
    """
    Apply Gaussian instrumental broadening. 
    This function broadens a spectrum assuming a Gaussian
    kernel. The width of the kernel is determined by the
    resolution. In particular, the function will determine
    the mean wavelength and set the Full Width at Half
    Maximum (FWHM) of the Gaussian to
    (mean wavelength)/resolution. 
    Parameters
    ----------
    wvl : array
        The wavelength
    flux : array
        The spectrum
    resolution : int
        The spectral resolution.
    edgeHandling : string, {None, "firstlast"}, optional
        Determines the way edges will be handled. If None,
        nothing will be done about it. If set to "firstlast",
        the spectrum will be extended by using the first and
        last value at the start or end. Note that this is
        not necessarily appropriate. The default is None.
    fullout : boolean, optional
        If True, also the FWHM of the Gaussian will be returned.
    maxsig : float, optional
        The extent of the broadening kernel in terms of
        standard deviations. By default, the Gaussian broadening
        kernel will be extended over the entire given spectrum,
        which can cause slow evaluation in the case of large spectra.
        A reasonable choice could, e.g., be five.  
    Returns
    -------
    Broadened spectrum : array
        The input spectrum convolved with a Gaussian
        kernel.
    FWHM : float, optional
        The Full Width at Half Maximum (FWHM) of the
        used Gaussian kernel.
    """
    # Check whether wvl axis is linear
    dwls = wvl[1:] - wvl[0:-1]
    #if abs(max(dwls) - min(dwls)) > np.mean(dwls) * 1e-6:
    #	print ('error: Las longitudes de onda deberian estar equiespaciadas')
    #    raise

    meanWvl = np.mean(wvl)
    fwhm = 1.0 / float(resolution) * meanWvl
    sigma = fwhm / (2.0 * np.sqrt(2. * np.log(2.)))

    e,x = broadGaussFast(
        wvl, flux, sigma, edgeHandling=edgeHandling, maxsig=maxsig)

    if not fullout:
        return result
    else:
        return e,x, fwhm




 ####leer IRAF spectra
def nonlinearwave(nwave, specstr, verbose=False):
    """Compute non-linear wavelengths from multispec string
    
    Returns wavelength array and dispersion fields.
    Raises a ValueError if it can't understand the dispersion string.
    """

    fields = specstr.split()
    if int(fields[2]) != 2:
        raise ValueError('Not nonlinear dispersion: dtype=' + fields[2])
    if len(fields) < 12:
        raise ValueError('Bad spectrum format (only %d fields)' % len(fields))
    wt = float(fields[9])
    w0 = float(fields[10])
    ftype = int(fields[11])
    if ftype == 3:

        # cubic spline

        if len(fields) < 15:
            raise ValueError('Bad spline format (only %d fields)' % len(fields))
        npieces = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            print 'Dispersion is order-%d cubic spline' % npieces
        if len(fields) != 15 + npieces + 3:
            raise ValueError('Bad order-%d spline format (%d fields)' % (npieces, len(fields)))
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        s = (np.arange(nwave, dtype=float) + 1 - pmin) / (pmax - pmin) * npieces
        j = s.astype(int).clip(0, npieces - 1)
        a = (j + 1) - s
        b = s - j
        x0 = a ** 3
        x1 = 1 + 3 * a * (1 + a * b)
        x2 = 1 + 3 * b * (1 + a * b)
        x3 = b ** 3
        wave = coeff[j] * x0 + coeff[j + 1] * x1 + coeff[j + 2] * x2 + coeff[j + 3] * x3

    elif ftype == 1 or ftype == 2:

        # chebyshev or legendre polynomial
        # legendre not tested yet

        if len(fields) < 15:
            raise ValueError('Bad polynomial format (only %d fields)' % len(fields))
        order = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if verbose:
            if ftype == 1:
                print 'Dispersion is order-%d Chebyshev polynomial' % order
            else:
                print 'Dispersion is order-%d Legendre polynomial (NEEDS TEST)' % order
        if len(fields) != 15 + order:
            # raise ValueError('Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
            if verbose:
                print 'Bad order-%d polynomial format (%d fields)' % (order, len(fields))
                print "Changing order from %i to %i" % (order, len(fields) - 15)
            order = len(fields) - 15
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        pmiddle = (pmax + pmin) / 2
        prange = pmax - pmin
        x = (np.arange(nwave, dtype=float) + 1 - pmiddle) / (prange / 2)
        p0 = np.ones(nwave, dtype=float)
        p1 = x
        wave = p0 * coeff[0] + p1 * coeff[1]
        for i in range(2, order):
            if ftype == 1:
                # chebyshev
                p2 = 2 * x * p1 - p0
            else:
                # legendre
                p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
            wave = wave + p2 * coeff[i]
            p0 = p1
            p1 = p2

    else:
        raise ValueError('Cannot handle dispersion function of type %d' % ftype)

    return wave, fields


def readmultispec(fitsfile, reform=True, quiet=False):
    """Read IRAF echelle spectrum in multispec format from a FITS file
    
    Can read most multispec formats including linear, log, cubic spline,
    Chebyshev or Legendre dispersion spectra
    
    If reform is true, a single spectrum dimensioned 4,1,NWAVE is returned
    as 4,NWAVE (this is the default.)  If reform is false, it is returned as
    a 3-D array.
    """

    fh = pyfits.open(fitsfile)
    try:
        header = fh[0].header
        flux = fh[0].data
    finally:
        fh.close()
    temp = flux.shape
    nwave = temp[-1]
    if len(temp) == 1:
        nspec = 1
    else:
        nspec = temp[-2]

    # first try linear dispersion
    try:
        crval1 = header['crval1']
        crpix1 = header['crpix1']
        cd1_1 = header['cd1_1']
        ctype1 = header['ctype1']
        if ctype1.strip() == 'LINEAR':
            wavelen = np.zeros((nspec, nwave), dtype=float)
            ww = (np.arange(nwave, dtype=float) + 1 - crpix1) * cd1_1 + crval1
            for i in range(nspec):
                wavelen[i, :] = ww
            # handle log spacing too
            dcflag = header.get('dc-flag', 0)
            if dcflag == 1:
                wavelen = 10.0 ** wavelen
                if not quiet:
                    print 'Dispersion is linear in log wavelength'
            elif dcflag == 0:
                if not quiet:
                    print 'Dispersion is linear'
            else:
                raise ValueError('Dispersion not linear or log (DC-FLAG=%s)' % dcflag)

            if nspec == 1 and reform:
                # get rid of unity dimensions
                flux = np.squeeze(flux)
                wavelen.shape = (nwave,)
            return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': None}
    except KeyError:
        pass

    # get wavelength parameters from multispec keywords
    try:
        wat2 = header['wat2_*']
        count = len(wat2)
    except KeyError:
        raise ValueError('Cannot decipher header, need either WAT2_ or CRVAL keywords')

    # concatenate them all together into one big string
    watstr = []
    for i in range(len(wat2)):
        # hack to fix the fact that older pyfits versions (< 3.1)
        # strip trailing blanks from string values in an apparently
        # irrecoverable way
        # v = wat2[i].value
        v = wat2[i]
        v = v + (" " * (68 - len(v)))  # restore trailing blanks
        watstr.append(v)
    watstr = ''.join(watstr)

    # find all the spec#="..." strings
    specstr = [''] * nspec
    for i in range(nspec):
        sname = 'spec' + str(i + 1)
        p1 = watstr.find(sname)
        p2 = watstr.find('"', p1)
        p3 = watstr.find('"', p2 + 1)
        if p1 < 0 or p1 < 0 or p3 < 0:
            raise ValueError('Cannot find ' + sname + ' in WAT2_* keyword')
        specstr[i] = watstr[p2 + 1:p3]

    wparms = np.zeros((nspec, 9), dtype=float)
    w1 = np.zeros(9, dtype=float)
    for i in range(nspec):
        w1 = np.asarray(specstr[i].split(), dtype=float)
        wparms[i, :] = w1[:9]
        if w1[2] == -1:
            raise ValueError('Spectrum %d has no wavelength calibration (type=%d)' %
                             (i + 1, w1[2]))
            # elif w1[6] != 0:
            #    raise ValueError('Spectrum %d has non-zero redshift (z=%f)' % (i+1,w1[6]))

    wavelen = np.zeros((nspec, nwave), dtype=float)
    wavefields = [None] * nspec
    for i in range(nspec):
        # if i in skipped_orders:
        #    continue
        verbose = (not quiet) and (i == 0)
        if wparms[i, 2] == 0 or wparms[i, 2] == 1:
            # simple linear or log spacing
            wavelen[i, :] = np.arange(nwave, dtype=float) * wparms[i, 4] + wparms[i, 3]
            if wparms[i, 2] == 1:
                wavelen[i, :] = 10.0 ** wavelen[i, :]
                if verbose:
                    print 'Dispersion is linear in log wavelength'
            elif verbose:
                print 'Dispersion is linear'
        else:
            # non-linear wavelengths
            wavelen[i, :], wavefields[i] = nonlinearwave(nwave, specstr[i],
                                                         verbose=verbose)
        wavelen *= 1.0 + wparms[i, 6]
        if verbose:
            print "Correcting for redshift: z=%f" % wparms[i, 6]
    if nspec == 1 and reform:
        # get rid of unity dimensions
        flux = np.squeeze(flux)
        wavelen.shape = (nwave,)
    return {'flux': flux, 'wavelen': wavelen, 'header': header, 'wavefields': wavefields}


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
parser.add_argument('--instrument', type=str, default='feros', help='espectrografo utilizado')
parser.add_argument('--skip', type=int, default=1, help='skip orders')

args = parser.parse_args()
spec  = args.spec
vel = args.vel
rango = args.rango
template = args.template
linstep = args.linstep
mode = args.mode

spectype = args.spectype
deconv = args.deconv
instrument = args.instrument
skip = args.skip


#leo espectros, los corrijo por velocidad
if instrument != 'tres':
	hdulist = pyfits.open(spec)
	hdulist.info()
	im = pyfits.getdata(spec)

	im[0,:,:] = shift_doppler(im[0,:,:],vel)
	#x = im [0,:,:]
	#y = im [1,:,:]
	lineal_x, lineal_y = linealizacion(im,linstep,mode)

else:
	spectra_list = read_fits.read_fits_spectrum1d(spec)

	#spectra_list = readmultispec(spec)
	#wav = spectra_list['wavelen']
	#flux = spectra_list['flux']
	#wav = wav +15.
	#wav = shift_doppler(wav,vel)
	lineal_x, lineal_y = linealizacion_tres(spectra_list,linstep,mode)
	#lineal_x = spectra_list['wavelen']
	#lineal_y = spectra_list['flux']




#leo template
if template == 't06000_g+2.5_m05p00_hr.fits':
	hmod = pyfits.getheader(template)
	smod = pyfits.getdata(template)[0]
	flux_syn = pyfits.getdata(template)[0]
	wav_syn = np.arange(len(smod))*hmod['CDELT1']+hmod['CRVAL1']
	wav_syn = ToVacuum(wav_syn)

elif template[:4] == 'ap00':

	data = np.loadtxt(template,usecols=[0,1],unpack=True)
	wav_syn = ToVacuum(data[0])
	flux_syn = data[1]


else:
	print 'por favor corregir nombre template'
	raise
#for i in np.arange(len(im[0,:,0])):#
	#im[0,i,:] = im[0,i,:][200:-200]#
	#im[3,i,:] = im[3,i,:][200:-200]
#	xa = im [0,i,:]#
#	ya = im [3,i,:]
#	plt.plot(xa,ya)
#plt.show()
#raise

#linealizo, puede ser con spline o con interpolacion lineal

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
wav,flux = merge(lineal_x,lineal_y)
#ordeno y limpio outliers
wav,flux = ordenar(wav,flux)
#wav,flux = clean_outliers(wav,flux)
wav,flux = ordenar(wav,flux)

#plt.plot(wav,flux)
#wav_syn = ToVacuum(wav_syn)

plt.plot(wav_syn,flux_syn)
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
#wav = ToVacuum(wav)
plt.plot(wav,flux)
plt.show()

#G,y = vsin_kernel(wav,flux,30.0,0.0)

#plt.plot(y,G)
#plt.show()

#e,x, fwhm = instrBroadGaussFast(wav, flux, 115000, edgeHandling="firstlast", fullout=True, maxsig=5.0)

#plt.plot(x,e)
#plt.show()

x = []
y = []
#lines = [6565,4863,4342] #5180 5380 5650 5850 6080 6480

for i in np.arange(num-skip):
	i = i +skip
	print 'rango numero',i
	print 'rango:',np.min(wav)+rango*(i),'to', np.min(wav)+rango*(1+i)
	#selecciono rango de 100 A
	mask = (wav > np.min(wav)+rango*(i) ) & (wav < np.min(wav)+rango*(1+i) )
	mask_syn = (wav_syn > np.min(wav)+rango*(i) ) & (wav_syn < np.min(wav)+rango*(1+i) )
	wav_aux = wav[mask]
	flux_aux = flux[mask]
	flux_syn_aux = flux_syn[mask_syn]
	wav_syn_aux = wav_syn[mask_syn]
	
	#reviso si hay lineas gruesas en el rango seleccionado, si hay, lo salto
	#if condicion_linea(wav_aux,) == True:
	#	continue
	#else:
	#	print 'en este rango no hay lineas gruesas'
	#plt.plot(wav_aux,flux_aux)
	#plt.plot(wav_aux,flux_aux,'ro')
	#plt.plot(wav_syn_aux,flux_syn_aux)
	#plt.plot(wav_syn_aux,flux_syn_aux,'bo')
	if np.sum(clean_lines(wav_aux,flux_aux,spectype)) < 50:
		print 'muy pocos puntos', i
		continue
	#corrijo con ajuste lineal, itero 5 para encontrar el mejor fit
	fitspec = fit_cleanlines_iter(wav_aux,flux_aux,1,5)
	fitsyn = fit_cleanlines_iter(wav_syn_aux,flux_syn_aux,1,5)
	#plt.plot(wav_aux,fitspec,'r')
	#plt.plot(wav_syn_aux,fitsyn,'g')
	#plt.show()
	dif = np.median(fitspec)-np.median(fitsyn)
	flux_aux = flux_aux-dif 

	#elimino lineas anchas
	#mask = clean_lines(wav_aux,flux_aux,spectype)
	#wav_aux = wav_aux[mask]
	#flux_aux = flux_aux[mask] 
	#mask = clean_lines(wav_syn_aux,flux_syn_aux,spectype)
	#flux_syn_aux = flux_syn_aux[mask] 
	#wav_syn_aux = wav_syn_aux[mask]
	wav_aux,flux_aux = ordenar(wav_aux,flux_aux)
	wav_syn_aux,flux_syn_aux = ordenar(wav_syn_aux,flux_syn_aux)
	print 'nans',np.sum((clean_lines(wav_aux,flux_aux,spectype)))
	#plt.plot(wav_aux,flux_aux)
	#plt.show()

	#xi,yi = rconvolution(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
	if deconv == 'vel':
		xi,yi = veldeconvolution2(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
	elif deconv == 'wav':
		xi,yi = wav_deconvolution(wav_aux,flux_aux,wav_syn_aux,flux_syn_aux)
	else:
		print 'error --decov: solo puede ser vel o wav'
		raise

	if len(xi) == 1 :
		continue
	#elimino deconvoluciones muy outliers
	#### defino arreglo velocidad. llevo los 2 a velocidades con el lambda central. Resampleo velocidades a velocidades preestablecidas
	#### veo que longitudes de onda tienen que ser con esas velocidades a ese lambda central 
	#### dopler a longitud de onda central obtengo arreglo de longitudes de onda. 
	#ajustar gaussina y ver que fit tiene mayor dispersion
	#gausfit = ajuste_gauss(xi,yi)
	#lim = int(len(yi)/10.*3)
	#short = np.std(yi[lim:-lim]/gausfit[lim:-lim])
	#plt.plot(xi,gausfit,'ro')
	#plt.plot(xi[lim:-lim],yi[lim:-lim]/gausfit[lim:-lim])
	#plt.plot(xi[mask],yi[mask])
	#plt.show()
	#short = np.std(yi[(xi < 0.2) & (xi > -0.2)])

	#print 'std',short

	#if short > 0.2:
	#	continue
	else:
		x.append(xi)
		y.append(yi)
		#plt.plot(xi,yi)
		#plt.show()

	#plot muestra mean de deconvolucion con i=8 rangos
	'''if i == 18:
		x = np.array(x)
		y = np.array(y)
		aux = []
		cont = np.arange(len(x))
		for i in np.arange(len(x[0])):
			clip = sigma_clip(y[:,i], sigma=1.5)    #aplico sigma clip
			aux = np.append(aux,np.asarray(cont)[clip.mask])

		auxx = collections.Counter(aux)
		print 'aux',aux 
		print 'auxx',auxx
		print 'comunes',auxx.most_common(3)
		print 'aux',auxx.most_common(3)[0]
		print 'aux',auxx.most_common(3)[0][0]

		raise'''
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
#plt.show()

print "termino deconvolucion"
#for i in np.arange(len(x)):
#	plt.plot(x[i],y[i])
#plt.show()

##las deconvoluciones tienen distinto largo (dif de 2 o 3 elementos)
##los dejo de la misma dimension
aux = []
for i in np.arange(len(y)):
	aux = np.append(aux,len(y[i]))
for j in np.arange(len(y)):
	y[j] = y[j][0:int(np.min(aux))]
	x[j] = x[j][0:int(np.min(aux))]

ymean = np.median(y,axis=0)
xmean = x[0].copy()

print "plot mean"
plt.plot(xmean,ymean,'bo')
#plt.show()

x = np.array(x)
y = np.array(y)
aux = []
cont = np.arange(len(x))
#mask = (x[0] < 50.) & (x[0] >-50)
'''for i in np.arange(len(x[0][mask])):
	if 
	clip = sigma_clip(y[:,i], sigma=1.5)    #aplico sigma clip
	aux = np.append(aux,np.asarray(cont)[clip.mask])
auxx = collections.Counter(aux)
print 'aux',aux 
print 'auxx',auxx
print 'comunes',auxx.most_common(3)
print 'aux',auxx.most_common(3)[0]
print 'aux',auxx.most_common(3)[0][0]'''

####limpio primera deconv 
n = len(x[0])

for i in np.arange(len(x[0][int(n/3):-int(n/3)])):
	clip = sigma_clip(y[:,i+int(n/3)], sigma=1.5)    #aplico sigma clip
	aux = np.append(aux,np.asarray(cont)[clip.mask])
auxx = collections.Counter(aux)
print 'aux',aux 
print 'auxx',auxx
print 'comunes',auxx.most_common(3)
print 'aux',auxx.most_common(3)[0]
print 'aux',auxx.most_common(3)[0][0]

n = 4
mask =  [True for i in range(len(x))] 
print mask
for i in np.arange(len(x)):
	for j in np.arange(len(auxx.most_common(n))):
		if int(auxx.most_common(n)[j][0]) == i:
			mask[i] = False
mask = np.array(mask)
cleanx = x[mask]
cleany = y[mask]

for i in np.arange(len(cleanx)):
	#plt.plot(cleanx[i][10:-10],cleany[i][10:-10],alpha=.3)
	plt.plot(cleanx[i],cleany[i],alpha=.7)

for i in np.arange(len(x[~mask])):
	#plt.plot(x[~mask][i][10:-10],y[~mask][i][10:-10],'k')
	plt.plot(x[~mask][i],y[~mask][i],'k',alpha=.5)

#plt.xlim(-80,80)
#plt.show()

ycleanmean = np.median(cleany,axis=0)
xcleanmean = cleanx[0].copy()

print "plot mean"
plt.plot(xcleanmean,ycleanmean,'bo')
plt.xlim(-80,80)
plt.show()



####aplico trapezoide

trapezoid_1D_kernel = Trapezoid1DKernel(2.0, slope=0.3)
ytrap = []
ytraplin = []
xtraplin = []

for i in np.arange(len(x)):
	auxy = np.convolve(y[i],trapezoid_1D_kernel,mode='same')
	ytrap.append(auxy)

	auxx = np.linspace(np.min(x[0]),np.max(x[0]),len(x[0])*5)
	tck = interpolate.splrep(x[i],ytrap[i],k=3)
	auxy2 = interpolate.splev(auxx,tck)
	ytraplin.append(auxy2)
	xtraplin.append(auxx)
	print ytrap[i]

	#Primera Deconv
	plt.plot(x[i],y[i],'k',alpha=.3)
	#Primera Deconv + trap
	plt.plot(x[i],ytrap[i])
	#Primera Deconv + trap , linealizado
	#plt.plot(xtraplin[i],ytraplin[i],'r',alpha=.3)

#plt.show()

#limpio trap 
xtraplin = np.array(xtraplin)
ytraplin = np.array(ytraplin)

n = len(xtraplin[0])
print 'n', n
for i in np.arange(len(xtraplin[0][int(n/3):-int(n/3)])):
	clip = sigma_clip(ytraplin[:,i+int(n/3)], sigma=1.5)    #aplico sigma clip
	aux = np.append(aux,np.asarray(cont)[clip.mask])
auxx = collections.Counter(aux)
print 'aux',aux 
print 'auxx',auxx
print 'comunes',auxx.most_common(3)
print 'aux',auxx.most_common(3)[0]
print 'aux',auxx.most_common(3)[0][0]

n = 4
mask =  [True for i in range(len(xtraplin))] 
print mask
for i in np.arange(len(xtraplin)):
	for j in np.arange(len(auxx.most_common(n))):
		if int(auxx.most_common(n)[j][0]) == i:
			mask[i] = False
mask = np.array(mask)

xtraplin_clean = xtraplin[mask]
ytraplin_clean = ytraplin[mask]

for i in np.arange(len(xtraplin_clean)):
	#plt.plot(cleanx[i][10:-10],cleany[i][10:-10],alpha=.3)
	plt.plot(xtraplin_clean[i],ytraplin_clean[i],alpha=.7)

for i in np.arange(len(xtraplin[~mask])):
	#plt.plot(x[~mask][i][10:-10],y[~mask][i][10:-10],'k')
	plt.plot(x[~mask][i],y[~mask][i],'k',alpha=.5)
plt.show()


#for i in np.arange(n):#
#	delet = auxx.most_common(n)[i][0]
#	newx = np.delete(newx, delet, axis=0)
####eliminar outliers, ver velocidad a velocidad, eliminar los que tengan mas velocidades malas
#for i in np.arange(3):
#	auxi = auxx.most_common(n+1)[i][0]
#	plt.plot(x[int(auxi)],y[int(auxi)]+0.1,'k')

#for i in np.arange(len(x)):#
#	plt.plot(x[i],y[i],'k')
#for i in np.arange(len(newx)):
#	plt.plot(newx[i],newy[i],alpha=.3)
#plt.show()

ytraplinclean_mean = np.median(ytraplin_clean,axis=0)
xtraplinclean_mean = xtraplin_clean[0].copy()

print "plot mean"
plt.plot(xtraplinclean_mean,ytraplinclean_mean)
plt.xlim(-80,80)
#plt.show()



ytrapmean = np.median(ytrap,axis=0)
xmean = cleanx[0].copy()

print "plot mean"
#plt.plot(xmean,ytrapmean)
#plt.xlim(-80,80)
#plt.show()

ytraplinmean = np.median(ytraplin,axis=0)
xtraplinmean = xtraplin[0].copy()

print "plot mean"
plt.plot(xtraplinmean,ytraplinmean,'r',alpha=.4)
plt.xlim(-80,80)
plt.show()


'''


xlin = []
ylin = []

for i in np.arange(len(cleanx)):
	auxx = np.linspace(np.min(cleanx[0]),np.max(cleanx[0]),len(cleanx[0])*5)
	#tck = interpolate.splrep(x,y,k=3)
	#fit = interpolate.splev(xdata,tck)

	tck = interpolate.splrep(cleanx[i],cleany[i],k=3)
	auxy = interpolate.splev(auxx,tck)
	xlin.append(auxx)
	ylin.append(auxy)
	#plt.plot(xlin[-1][10:-10],ylin[-1][10:-10])
	plt.plot(xlin[-1],ylin[-1])

plt.xlim(-80,80)
plt.show()

ynewmean = np.median(ylin,axis=0)
xnewmean = xlin[0].copy()
print ('empieza condiciones')

trapezoid_1D_kernel = Trapezoid1DKernel(1.2, slope=0.2)
ylintrap = []
for i in np.arange(len(xlin)):
	auxy = np.convolve(ylin[i],trapezoid_1D_kernel,mode='same')
	ylintrap.append(auxy)
	plt.plot(xlin[i],ylin[i],'k',alpha=.3)
	plt.plot(xlin[i],ylintrap[i])
plt.show()

if (len(xnew[0]) > len(ynewmean)):
	xnewmean = xnewmean[0:len(ynewmean)]
elif (len(x[0]) < len(ymean)):
	ynewmean = ynewmean[0:len(x[0])]

#ymean = np.mean(y,axis=0)

#plt.plot(xnewmean[20:-20],ynewmean[20:-20])
#plt.xlim(-50,50)
plt.plot(xnewmean,ynewmean)
plt.xlim(-80,80)

plt.show()



trapezoid_1D_kernel = Trapezoid1DKernel(1, slope=0.2)
yfiltered = np.convolve(ynewmean,trapezoid_1D_kernel,mode='same')
plt.plot(xnewmean,yfiltered)
plt.show()


raise
print ('empieza condiciones')
if (len(x[0]) > len(ymean)):
	print 'se cumple 1'
	xmean = xmean[0:len(ymean)]
elif (len(x[0]) < len(ymean)):
	print 'se cumple 1'

	ymean = ymean[0:len(x[0])]
else:
	'no se cumple nada'
#ymean = np.mean(y,axis=0)



'''



