import os
import numpy as np
import matplotlib
import time
from sklearn import preprocessing, cross_validation, neighbors, svm, tree 
from sklearn.model_selection import train_test_split
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import argparse
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd

path = 'C:\Users\Mahesh\Anaconda2\envs\MBET\DATA\METRO'
files =  os.listdir(path)

##Test Start##
## Calling the actuall file ##
#allmetrodata = open("testfile.csv","r")

#x = allmetrodata.readlines()


#final =[]
#result = []

##Used ot split the data intuitivly based on the elements##
#for i in x:
#	result.append(i.split(","))
#print result

## The different variables and their index position ##
## 0.Time
## 1.Latitude
## 2.Longitude
## 3.Speed
## 4.Accuracy
## 5.Bearing
## 6.Label
## 7.X
## 8.Y
## 9.Z


#######################################
## some random testing that i cant 
## remember why i do that
#array =  np.array(x)
#print result

#lsttest = [item[2] for item in result]
#F = map(float, lsttest)
########################################

### This will be the amount of times i want the definition fuctions to 
### iterate over the lists ###
n=3000
########################################

######## All definitions functions Below ##########

#variance = np.var(F, dtype=np.float64)
def variance(x):
	np.var(x,dtype=np.float64)
#print "the variance is " + str(variance)
#print "The Max is " + str(max(F))
#print "The min is " + str(min(F))

#Average =  sum(F)/float(len(F))
def average(x):
	sum(x)/float(len(x))
#print "the average is " + str(Average)

#Gradient = np.gradient(F)
def gradient(x):
	return np.gradient(x)
#print "The gradient is " + str(Gradient)

#Minlist =[min(F) for i in range(0,len(F),n)] 
def minlist(x):
	output = []
	index = 0
	while index != len(x):
		if index % n != 0:
			output.append(min(x[index:len(x)]))
			#print output
		if x[index:index+n] == []:
			#print output
			return output
		step1 = min(x[index:index+n])
		output.append(step1)
		index = index + n
	return output
	#return [min(x) for i in range(0,len(x),n)] 

##TEST MIN LIST##
print minlist([1,3,2,5,4,7,1,3,2,5,4,7,1,2,5,4,7,1,3,2,5,4,7])

#Maxlist = [max(F) for i in range(0,len(F),n)]
def maxlist(x):
	output = []
	index = 0
	while index != len(x):
		if index % n != 0:
			output.append(max(x[index:len(x)]))
			#print output
		if x[index:index+n] == []:
			#print output
			return output
		step1 = max(x[index:index+n])
		output.append(step1)
		index = index + n
	print output
	#return [max(x) for i in range(0,len(x),n)]

#TEST MAX LIST##
maxlist([1,3,2,5,4,7,1,3,2,5,4,7,1,2,5,4,7,1,3,2,5,4,7])

#AvgList = [(sum(F)/len(F)) for i in range(0,len(F),n)]
def avglist(x):
	output = []
	index = 0
	while index != len(x):
		if index % n != 0:
			output.append(sum(x[index:len(x)])/len(x[index:index+n]))
			#print output
		if x[index:index+n] == []:
			#print output
			return output
		step1 = sum(x[index:index+n])/len(x[index:index+n])
		output.append(step1)
		index = index + n
	print output
	#return [(sum(x)/len(x)) for i in range(0,len(x),n)]

##TEST FOR AVGLIST##
avglist([1,3,2,5,4,7,1,3,2,5,4,7,1,2,5,4,7,1,3,2,5,4,7])


#VarianceList = [(np.var(F)) for i in range(0,len(F),n)] 
def variancelist(x):
	output = []
	index = 0
	while index != len(x):
		if index % n != 0:
			output.append(np.var(x[index:len(x)]))
			#print output
		if x[index:index+n] == []:
			#print output
			return output
		step1 = np.var(x[index:index+n])
		output.append(step1)
		index = index + n
	print output
	#return [(np.var(x)) for i in range(0,len(x),n)] 

##TEST FRO variancelist##
variancelist([1,3,2,5,4,7,1,3,2,5,4,7,1,2,5,4,7,1,3,2,5,4,7])

#GradientList = [(np.gradient(F)) for i in range(0,len(F),n)]
def gradientlist(x):
	output = []
	index = 0
	while index != len(x):
		if index % n != 0:
			output.append(np.gradient(x[index:len(x)]))
			#print output
		if x[index:index+n] == []:
			return output
		step1 = np.gradient(x[index:index+n])
		output.append(step1)
		index = index + n
	return output
	#return [(np.gradient(x)) for i in range(0,len(x),n)]

##TEST FOR GRADIENTLIST##
#gradientlist([1,3,2,5,4,7,1,3,2,5,4,7,1,2,5,4,7,1,3,2,5,4,7])

###############################################################

### All columns as their own list ###
## 0.CompressedTime

# lst0 = [item[0] for item in result]
# CompressedTime = map(float, lst0)

# ## 1.CompressedLatitude
# lst1 = [item[1] for item in result]
# CompressedLatitude = map(float, lst1)

# ## 2.CompressedLongitude
# lst2 = [item[2] for item in result]
# CompressedLongitude = map(float, lst2)

# ## 3.CompressedSpeed
# lst3 = [item[3] for item in result]
# compressedSpeed= map(float, lst3)
# #print compressedSpeed
# ## 4.CompressedAccuracy
# lst4 = [item[4] for item in result]
# CompressedAccuracy = map(float, lst4)

# ## 5.CompressedBearing
# lst5 = [item[5] for item in result]
# CompressedBearing = map(float, lst5)

# ## 6.CompressedLabel
# lst6 = [item[6] for item in result]
# CompressedLabel = map(float, lst6)

# ## 7.CompressedX
# lst7 = [item[7] for item in result]
# CompressedX = map(float, lst7)
# ## 8.CompressedY
# lst8 = [item[8] for item in result]
# CompressedY = map(float, lst8)
# ## 9.CompressedZ
# lst9 = [item[9] for item in result]
# CompressedZ = map(float, lst9)

###########################################################

######################### Lists used for the cross product ####################

#Variables = [CompressedTime, CompressedLatitude, CompressedLongitude, compressedSpeed, CompressedAccuracy, CompressedBearing, CompressedLabel, CompressedX, CompressedY, CompressedZ]
#processLst = [variancelist, maxlist, minlist, avglist]
###############################################################################

def avg():
	pass


print ""
print ""


#print variancelist(CompressedY)[:1000]

#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
def removedot(a):
	final =[]
	for lst in a:
		final.append(lst)
		countofa = 0
		for i in lst:
			if countofa >= 2:
				break
			countofa = 0
			for element in i:
				if element == ".":
					countofa += 1
				if countofa >= 2:
					final = final[:-1]
					break
	return final



################################################################################
### Change all foot to transportation mode in file of csvreader###

######################################
## Foot Data ##
import csv
csvreaderfoot = csv.reader(open("C:\Users\Mahesh\Anaconda2\envs\MBET\DATA\FOOT_WALKING\ALLFOOTDATA.csv"))
#print csvreaderfoot

footTimeArray = []
footLatArray = []
footLonArray = []
footspeedArray =[]
footaccuracyArray=[]
footbearingArray=[]
footlabelArray = []
footXArray =[]
footYArray =[]
footZArray =[]
# Add more feature if you want


for row in csvreaderfoot:
	if len(row) == 10:
		footTimeArray.append(float(row[0]))
		footLatArray.append(float(row[1]))
		footLonArray.append(float(row[2]))
		footspeedArray.append(float(row[3]))
		footaccuracyArray.append(float(row[4]))
		footbearingArray.append(float(row[5]))
		footlabelArray.append(row[6])
		footXArray.append(float(row[7]))
		footYArray.append(float(row[8]))
		footZArray.append(float(row[9]))
	#add other feature if you want

#convert array to numpy array
foot_time = np.array(footTimeArray)
foot_lat = np.array(footLatArray)
foot_lon = np.array(footLonArray)
foot_speed = np.array(footspeedArray)
foot_accuracy = np.array(footaccuracyArray)
foot_bearing = np.array(footbearingArray)
foot_label= np.array(footlabelArray)
foot_X = np.array(footXArray)
foot_Y = np.array(footYArray)
foot_Z = np.array(footZArray)

##### avergares
foot_TimeArrayAVG = np.array(avglist(footTimeArray))
foot_LatArrayAVG = np.array(avglist(footLatArray))
foot_LonArrayAVG = np.array(avglist(footLonArray))
foot_SpeedArrayAVG = np.array(avglist(footspeedArray))
foot_AccuracyArrayAVG = np.array(avglist(footaccuracyArray))
foot_BearingArrayAVG = np.array(avglist(footbearingArray))
foot_LabelArrayAVG = np.array(avglist(map(int,footlabelArray)))
foot_XArrayAVG = np.array(avglist(footXArray))
foot_YArrayAVG = np.array(avglist(footYArray))
foot_ZArrayAVG = np.array(avglist(footZArray))
##### maxes
foot_TimeArrayMAX = np.array(maxlist(footTimeArray))
foot_LatArrayMAX = np.array(maxlist(footLatArray))
foot_LonArrayMAX = np.array(maxlist(footLonArray))
foot_SpeedArrayMAX = np.array(maxlist(footspeedArray))
foot_AccuracyArrayMAX = np.array(maxlist(footaccuracyArray))
foot_BearingArrayMAX = np.array(maxlist(footbearingArray))
foot_LabelArrayMAX = np.array(maxlist(map(int,footlabelArray)))
foot_XArrayMAX = np.array(maxlist(footXArray))
foot_YArrayMAX = np.array(maxlist(footYArray))
foot_ZArrayMAX = np.array(maxlist(footZArray))
##### mins
foot_TimeArrayMIN = np.array(minlist(footTimeArray))
foot_LatArrayMIN = np.array(minlist(footLatArray))
foot_LonArrayMIN = np.array(minlist(footLonArray))
foot_SpeedArrayMIN = np.array(minlist(footspeedArray))
foot_AccuracyArrayMIN = np.array(minlist(footaccuracyArray))
foot_BearingArrayMIN = np.array(minlist(footbearingArray))
foot_LabelArrayMIN = np.array(minlist(map(int,footlabelArray)))
foot_XArrayMIN = np.array(minlist(footXArray))
foot_YArrayMIN = np.array(minlist(footYArray))
foot_ZArrayMIN = np.array(minlist(footZArray))

allFootData = np.column_stack([foot_SpeedArrayMIN, foot_SpeedArrayMAX, foot_SpeedArrayAVG, 
			foot_AccuracyArrayMIN, foot_AccuracyArrayMAX, foot_AccuracyArrayAVG, 
			foot_BearingArrayMIN, foot_BearingArrayMAX, foot_BearingArrayAVG, 
			foot_XArrayMIN, foot_XArrayMAX, foot_XArrayAVG, 
			foot_YArrayMIN,foot_YArrayMAX, foot_YArrayAVG, 
			foot_ZArrayMIN, foot_ZArrayMAX, foot_ZArrayAVG])
#allFootData = np.column_stack([foot_speed, foot_accuracy, foot_bearing, foot_X, foot_Y, foot_Z])
print "all foot data shape ", allFootData.shape
########################################
#################car data##############################
csvreadercar = csv.reader(open("C:\Users\Mahesh\Anaconda2\envs\MBET\DATA\CARRO\AllCarro.csv"))

carTimeArray = []
carLatArray = []
carLonArray = []
carspeedArray =[]
caraccuracyArray=[]
carbearingArray=[]
carlabelArray = []
carXArray =[]
carYArray =[]
carZArray =[]

for row in csvreadercar:
	if len(row) == 10:
		carTimeArray.append(float(row[0]))
		carLatArray.append(float(row[1]))
		carLonArray.append(float(row[2]))
		carspeedArray.append(float(row[3]))
		caraccuracyArray.append(float(row[4]))
		carbearingArray.append(float(row[5]))
		carlabelArray.append(row[6])
		carXArray.append(float(row[7]))
		carYArray.append(float(row[8]))
		carZArray.append(float(row[9]))

car_time = np.array(carTimeArray)
car_lat = np.array(carLatArray)
car_lon = np.array(carLonArray)
car_speed = np.array(carspeedArray)
car_accuracy = np.array(caraccuracyArray)
car_bearing = np.array(carbearingArray)
car_label= np.array(carlabelArray)
car_X = np.array(carXArray)
car_Y = np.array(carYArray)
car_Z = np.array(carZArray)

##### avergares
car_TimeArrayAVG = np.array(avglist(carTimeArray))
car_LatArrayAVG = np.array(avglist(carLatArray))
car_LonArrayAVG = np.array(avglist(carLonArray))
car_SpeedArrayAVG = np.array(avglist(carspeedArray))
car_AccuracyArrayAVG = np.array(avglist(caraccuracyArray))
car_BearingArrayAVG = np.array(avglist(carbearingArray))
car_LabelArrayAVG = np.array(avglist(map(int,carlabelArray)))
car_XArrayAVG = np.array(avglist(carXArray))
car_YArrayAVG = np.array(avglist(carYArray))
car_ZArrayAVG = np.array(avglist(carZArray))
##### maxes
car_TimeArrayMAX = np.array(maxlist(carTimeArray))
car_LatArrayMAX = np.array(maxlist(carLatArray))
car_LonArrayMAX = np.array(maxlist(carLonArray))
car_SpeedArrayMAX = np.array(maxlist(carspeedArray))
car_AccuracyArrayMAX = np.array(maxlist(caraccuracyArray))
car_BearingArrayMAX = np.array(maxlist(carbearingArray))
car_LabelArrayMAX = np.array(maxlist(map(int,carlabelArray)))
car_XArrayMAX = np.array(maxlist(carXArray))
car_YArrayMAX = np.array(maxlist(carYArray))
car_ZArrayMAX = np.array(maxlist(carZArray))
##### mins
car_TimeArrayMIN = np.array(minlist(carTimeArray))
car_LatArrayMIN = np.array(minlist(carLatArray))
car_LonArrayMIN = np.array(minlist(carLonArray))
car_SpeedArrayMIN = np.array(minlist(carspeedArray))
car_AccuracyArrayMIN = np.array(minlist(caraccuracyArray))
car_BearingArrayMIN = np.array(minlist(carbearingArray))
car_LabelArrayMIN = np.array(minlist(map(int,carlabelArray)))
car_XArrayMIN = np.array(minlist(carXArray))
car_YArrayMIN = np.array(minlist(carYArray))
car_ZArrayMIN = np.array(minlist(carZArray))

allcarData = np.column_stack([car_SpeedArrayMIN, car_SpeedArrayMAX, car_SpeedArrayAVG, 
			car_AccuracyArrayMIN, car_AccuracyArrayMAX, car_AccuracyArrayAVG, 
			car_BearingArrayMIN, car_BearingArrayMAX, car_BearingArrayAVG,
			car_XArrayMIN, car_XArrayMAX, car_XArrayAVG, 
			car_YArrayMIN, car_YArrayMAX, car_YArrayAVG, 
			car_ZArrayMIN, car_ZArrayMAX, car_ZArrayAVG])

print "the shape of the car is", allcarData.shape

######################################
### bus data ###

csvreaderbus = removedot(csv.reader(open("C:\Users\Mahesh\Anaconda2\envs\MBET\DATA\BUS\AllBusFile.csv")))
#print csvreaderbus

busTimeArray = []
busLatArray = []
busLonArray = []
busspeedArray =[]
busaccuracyArray=[]
busbearingArray=[]
buslabelArray = []
busXArray =[]
busYArray =[]
busZArray =[]
# Add more feature if you want

for row in csvreaderbus:
	row = row[0:10]
	if all(row) == True:
		busTimeArray.append(float(row[0])) 
		busLatArray.append(float(row[1]))
		busLonArray.append(float(row[2]))
		busspeedArray.append(float(row[3]))
		busaccuracyArray.append(float(row[4]))
		busbearingArray.append(float(row[5]))
		buslabelArray.append("0")
		busXArray.append(float(row[7]))
		busYArray.append(float(row[8]))
		busZArray.append(float(row[9]))
	else:
		print row
	#add other feature if you want

#convert array to numpy array
bus_time = np.array(busTimeArray)
bus_lat = np.array(busLatArray)
bus_lon = np.array(busLonArray)
bus_speed = np.array(busspeedArray)
bus_accuracy = np.array(busaccuracyArray)
bus_bearing = np.array(busbearingArray)
bus_label= np.array(buslabelArray)
bus_X = np.array(busXArray)
bus_Y = np.array(busYArray)
bus_Z = np.array(busZArray)

##### avergares
bus_TimeArrayAVG = np.array(avglist(busTimeArray))
bus_LatArrayAVG = np.array(avglist(busLatArray))
bus_LonArrayAVG = np.array(avglist(busLonArray))
bus_SpeedArrayAVG = np.array(avglist(busspeedArray))
bus_AccuracyArrayAVG = np.array(avglist(busaccuracyArray))
bus_BearingArrayAVG = np.array(avglist(busbearingArray))
bus_LabelArrayAVG = np.array(avglist(map(int,buslabelArray)))
bus_XArrayAVG = np.array(avglist(busXArray))
bus_YArrayAVG = np.array(avglist(busYArray))
bus_ZArrayAVG = np.array(avglist(busZArray))
##### maxes
bus_TimeArrayMAX = np.array(maxlist(busTimeArray))
bus_LatArrayMAX = np.array(maxlist(busLatArray))
bus_LonArrayMAX = np.array(maxlist(busLonArray))
bus_SpeedArrayMAX = np.array(maxlist(busspeedArray))
bus_AccuracyArrayMAX = np.array(maxlist(busaccuracyArray))
bus_BearingArrayMAX = np.array(maxlist(busbearingArray))
bus_LabelArrayMAX = np.array(maxlist(map(int,buslabelArray)))
bus_XArrayMAX = np.array(maxlist(busXArray))
bus_YArrayMAX = np.array(maxlist(busYArray))
bus_ZArrayMAX = np.array(maxlist(busZArray))
##### mins
bus_TimeArrayMIN = np.array(minlist(busTimeArray))
bus_LatArrayMIN = np.array(minlist(busLatArray))
bus_LonArrayMIN = np.array(minlist(busLonArray))
bus_SpeedArrayMIN = np.array(minlist(busspeedArray))
bus_AccuracyArrayMIN = np.array(minlist(busaccuracyArray))
bus_BearingArrayMIN = np.array(minlist(busbearingArray))
bus_LabelArrayMIN = np.array(minlist(map(int,buslabelArray)))
bus_XArrayMIN = np.array(minlist(busXArray))
bus_YArrayMIN = np.array(minlist(busYArray))
bus_ZArrayMIN = np.array(minlist(busZArray))

allbusData = np.column_stack([bus_SpeedArrayMIN, bus_SpeedArrayMAX, bus_SpeedArrayAVG, 
			bus_AccuracyArrayMIN, bus_AccuracyArrayMAX, bus_AccuracyArrayAVG, 
			bus_BearingArrayMIN, bus_BearingArrayMAX, bus_BearingArrayAVG, 
			bus_XArrayMIN, bus_XArrayMAX, bus_XArrayAVG, 
			bus_YArrayMIN, bus_YArrayMAX, bus_YArrayAVG, 
			bus_ZArrayMIN, bus_ZArrayMAX, bus_ZArrayAVG])
#allbusData = np.column_stack([bus_speed, bus_accuracy, bus_bearing, bus_X, bus_Y, bus_Z])
#print allbusData[10:100]
print "all bus data shape ", allbusData.shape


## All Metro Data
print "strat metro preprocessing"
#########################################################
csvreadermetro = csv.reader(open("C:\Users\Mahesh\Anaconda2\envs\MBET\DATA\METRO\Allmetrodata.csv"))
#print csvreadermetro

metroTimeArray = []
metroLatArray = []
metroLonArray = []
metrospeedArray =[]
metroaccuracyArray=[]
metrobearingArray=[]
metrolabelArray = []
metroXArray =[]
metroYArray =[]
metroZArray =[]
# Add more feature if you want


for row in csvreadermetro:
	if len(row) == 10:
		metroTimeArray.append(float(row[0]))
		metroLatArray.append(float(row[1]))
		metroLonArray.append(float(row[2]))
		metrospeedArray.append(float(row[3]))
		metroaccuracyArray.append(float(row[4]))
		metrobearingArray.append(float(row[5]))
		metrolabelArray.append(row[6])
		metroXArray.append(float(row[7]))
		metroYArray.append(float(row[8]))
		metroZArray.append(float(row[9]))
	#add other feature if you want

#convert array to numpy array
metro_time = np.array(metroTimeArray)
metro_lat = np.array(metroLatArray)
metro_lon = np.array(metroLonArray)
metro_speed = np.array(metrospeedArray)
metro_accuracy = np.array(metroaccuracyArray)
metro_bearing = np.array(metrobearingArray)
metro_label= np.array(metrolabelArray)
metro_X = np.array(metroXArray)
metro_Y = np.array(metroYArray)
metro_Z = np.array(metroZArray)

##### avergares
metro_TimeArrayAVG = np.array(avglist(metroTimeArray))
metro_LatArrayAVG = np.array(avglist(metroLatArray))
metro_LonArrayAVG = np.array(avglist(metroLonArray))
metro_SpeedArrayAVG = np.array(avglist(metrospeedArray))
metro_AccuracyArrayAVG = np.array(avglist(metroaccuracyArray))
metro_BearingArrayAVG = np.array(avglist(metrobearingArray))
metro_LabelArrayAVG = np.array(avglist(map(int,metrolabelArray)))
metro_XArrayAVG = np.array(avglist(metroXArray))
metro_YArrayAVG = np.array(avglist(metroYArray))
metro_ZArrayAVG = np.array(avglist(metroZArray))
##### maxes
metro_TimeArrayMAX = np.array(maxlist(metroTimeArray))
metro_LatArrayMAX = np.array(maxlist(metroLatArray))
metro_LonArrayMAX = np.array(maxlist(metroLonArray))
metro_SpeedArrayMAX = np.array(maxlist(metrospeedArray))
metro_AccuracyArrayMAX = np.array(maxlist(metroaccuracyArray))
metro_BearingArrayMAX = np.array(maxlist(metrobearingArray))
metro_LabelArrayMAX = np.array(maxlist(map(int,metrolabelArray)))
metro_XArrayMAX = np.array(maxlist(metroXArray))
metro_YArrayMAX = np.array(maxlist(metroYArray))
metro_ZArrayMAX = np.array(maxlist(metroZArray))
##### mins
metro_TimeArrayMIN = np.array(minlist(metroTimeArray))
metro_LatArrayMIN = np.array(minlist(metroLatArray))
metro_LonArrayMIN = np.array(minlist(metroLonArray))
metro_SpeedArrayMIN = np.array(minlist(metrospeedArray))
metro_AccuracyArrayMIN = np.array(minlist(metroaccuracyArray))
metro_BearingArrayMIN = np.array(minlist(metrobearingArray))
metro_LabelArrayMIN = np.array(minlist(map(int,metrolabelArray)))
metro_XArrayMIN = np.array(minlist(metroXArray))
metro_YArrayMIN = np.array(minlist(metroYArray))
metro_ZArrayMIN = np.array(minlist(metroZArray))

allmetroData = np.column_stack([metro_SpeedArrayMIN, metro_SpeedArrayMAX, metro_SpeedArrayAVG, 
			metro_AccuracyArrayMIN, metro_AccuracyArrayMAX, metro_AccuracyArrayAVG, 
			metro_BearingArrayMIN, metro_BearingArrayMAX, metro_BearingArrayAVG, 
			metro_XArrayMIN, metro_XArrayMAX, metro_XArrayAVG, 
			metro_YArrayMIN,metro_YArrayMAX, metro_YArrayAVG, 
			metro_ZArrayMIN, metro_ZArrayMAX, metro_ZArrayAVG])
#allmetroData = np.column_stack([metro_speed, metro_accuracy, metro_bearing, metro_X, metro_Y, metro_Z])
print "all metro data shape ", allmetroData.shape


#########################################################
## All Running Data
print "start running preprocessing"
csvreaderrunning = csv.reader(open("C:\Users\Mahesh\Anaconda2\envs\MBET\DATA\RUNNING\AllRunningData.csv"))
#print csvreaderrunning

runningTimeArray = []
runningLatArray = []
runningLonArray = []
runningspeedArray =[]
runningaccuracyArray=[]
runningbearingArray=[]
runninglabelArray = []
runningXArray =[]
runningYArray =[]
runningZArray =[]
#Add more feature if you want

for row in csvreaderrunning:
	if len(row) == 9:
		runningTimeArray.append(float(row[0].replace(':','')))
		runningLatArray.append(float(row[1]))
		runningLonArray.append(float(row[2]))
		runningspeedArray.append(float(row[3]))
		runningaccuracyArray.append(float(row[4]))
		#runningbearingArray.append(float(row[5]))
		runninglabelArray.append(row[5])
		runningXArray.append(float(row[6]))
		runningYArray.append(float(row[7]))
		runningZArray.append(float(row[8]))
	#add other feature if you want

#convert array to numpy array
running_time = np.array(runningTimeArray)
running_lat = np.array(runningLatArray)
running_lon = np.array(runningLonArray)
running_speed = np.array(runningspeedArray)
running_accuracy = np.array(runningaccuracyArray)
#runnning_bearing = np.array(runningbearingArray)
running_label= np.array(runninglabelArray)
running_X = np.array(runningXArray)
running_Y = np.array(runningYArray)
running_Z = np.array(runningZArray)

#allrunningData = np.column_stack([running_lat, running_lon, running_speed, running_accuracy, running_bearing, running_X, running_Y, running_Z])
#allrunningData = np.column_stack([running_speed, running_accuracy, running_bearing, running_X, running_Y, running_Z])
#print "all running data shape ", allrunningData.shape


## All Subway Data


## All Trem Data
print "start running preprocessing"
csvreadertrem = csv.reader(open("C:\Users\Mahesh\Anaconda2\envs\MBET\DATA\TREM\Alltremdata.csv"))
#print csvreaderrunning

tremTimeArray = []
tremLatArray = []
tremLonArray = []
tremspeedArray =[]
tremaccuracyArray=[]
trembearingArray=[]
tremlabelArray = []
tremXArray =[]
tremYArray =[]
tremZArray =[]
#Add more feature if you want

for row in csvreadertrem:
	if len(row) == 10:
		tremTimeArray.append(float(row[0].replace(':','')))
		tremLatArray.append(float(row[1]))
		tremLonArray.append(float(row[2]))
		tremspeedArray.append(float(row[3]))
		tremaccuracyArray.append(float(row[4]))
		trembearingArray.append(float(row[5]))
		tremlabelArray.append(row[6])
		tremXArray.append(float(row[7]))
		tremYArray.append(float(row[8]))
		tremZArray.append(float(row[9]))
	#add other feature if you want

#convert array to numpy array
trem_time = np.array(tremTimeArray)
trem_lat = np.array(tremLatArray)
trem_lon = np.array(tremLonArray)
trem_speed = np.array(tremspeedArray)
trem_accuracy = np.array(tremaccuracyArray)
trem_bearing = np.array(trembearingArray)
trem_label= np.array(tremlabelArray)
trem_X = np.array(tremXArray)
trem_Y = np.array(tremYArray)
trem_Z = np.array(tremZArray)

##### avergares
trem_TimeArrayAVG = np.array(avglist(tremTimeArray))
trem_LatArrayAVG = np.array(avglist(tremLatArray))
trem_LonArrayAVG = np.array(avglist(tremLonArray))
trem_SpeedArrayAVG = np.array(avglist(tremspeedArray))
trem_AccuracyArrayAVG = np.array(avglist(tremaccuracyArray))
trem_BearingArrayAVG = np.array(avglist(trembearingArray))
trem_LabelArrayAVG = np.array(avglist(map(int,tremlabelArray)))
trem_XArrayAVG = np.array(avglist(tremXArray))
trem_YArrayAVG = np.array(avglist(tremYArray))
trem_ZArrayAVG = np.array(avglist(tremZArray))
##### maxes
trem_TimeArrayMAX = np.array(maxlist(tremTimeArray))
trem_LatArrayMAX = np.array(maxlist(tremLatArray))
trem_LonArrayMAX = np.array(maxlist(tremLonArray))
trem_SpeedArrayMAX = np.array(maxlist(tremspeedArray))
trem_AccuracyArrayMAX = np.array(maxlist(tremaccuracyArray))
trem_BearingArrayMAX = np.array(maxlist(trembearingArray))
trem_LabelArrayMAX = np.array(maxlist(map(int,tremlabelArray)))
trem_XArrayMAX = np.array(maxlist(tremXArray))
trem_YArrayMAX = np.array(maxlist(tremYArray))
trem_ZArrayMAX = np.array(maxlist(tremZArray))
##### mins
trem_TimeArrayMIN = np.array(minlist(tremTimeArray))
trem_LatArrayMIN = np.array(minlist(tremLatArray))
trem_LonArrayMIN = np.array(minlist(tremLonArray))
trem_SpeedArrayMIN = np.array(minlist(tremspeedArray))
trem_AccuracyArrayMIN = np.array(minlist(tremaccuracyArray))
trem_BearingArrayMIN = np.array(minlist(trembearingArray))
trem_LabelArrayMIN = np.array(minlist(map(int,tremlabelArray)))
trem_XArrayMIN = np.array(minlist(tremXArray))
trem_YArrayMIN = np.array(minlist(tremYArray))
trem_ZArrayMIN = np.array(minlist(tremZArray))

alltremData = np.column_stack([trem_SpeedArrayMIN, trem_SpeedArrayMAX, trem_SpeedArrayAVG, 
			trem_AccuracyArrayMIN, trem_AccuracyArrayMAX, trem_AccuracyArrayAVG, 
			trem_BearingArrayMIN, trem_BearingArrayMAX, trem_BearingArrayAVG, 
			trem_XArrayMIN, trem_XArrayMAX, trem_XArrayAVG, 
			trem_YArrayMIN, trem_YArrayMAX, trem_YArrayAVG, 
			trem_ZArrayMIN, trem_ZArrayMAX, trem_ZArrayAVG])
#allrunningData = np.column_stack([running_speed, running_accuracy, running_bearing, running_X, running_Y, running_Z])
print "all trem data shape ", alltremData.shape


### Combine all the data ####
#Consolidate all data#
############################
##############################
all_data = np.concatenate((allFootData, allbusData, allcarData, allmetroData, alltremData),axis=0)
all_labels = np.array(map(float, np.concatenate((foot_LabelArrayAVG, bus_LabelArrayAVG, car_LabelArrayAVG, metro_LabelArrayAVG, trem_LabelArrayAVG), axis=0)))

#print "Start printing all data"
#print all_data
#print "End printing all data"
###########################
#######################$##
#runfunction (all_data,n) => 

print "all data shape", all_data.shape
#print all_data
print "all labels shape", all_labels.shape
#print all_labels

#############################################################################
#############################################################################

## Actual cross product##
#print "start compressed list"
#print max(bus_speed)
#print maxlist(bus_speed)
# compressedlst = []
# for process in processLst:
# 	for lst in [bus_speed, bus_accuracy, bus_bearing, bus_X, bus_Y, bus_Z]:
# 		compressedlst.append(process(lst))
# print len(compressedlst)
#print "end compressed list"

#############################################################################
#############################################################################
####Classification Time###
Data_train, Data_test, Labels_train, Labels_test = cross_validation.train_test_split(all_data, all_labels, test_size=0.2)
X = Data_train
y = map(int,Labels_train)

k_best_selector = SelectKBest(f_regression, k=5)

################ different classifiers##############
# classifier = ExtraTreesClassifier(n_estimators=60, max_depth=3)
classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)
print "gaussian classifier is ",y_pred
print y[:10] ==y_pred[:10] 
accuracy = 100.0*(y == y_pred).sum() / X.shape[0]
print "gaussian accuracy is ", accuracy

processor_pipeline = Pipeline([('selector', k_best_selector),('rf', classifier)])
processor_pipeline.fit(X,y)
print "pipeline predict is ",processor_pipeline.predict(X)
print "pipeline score is ",processor_pipeline.score(X,y)

print "cross_validation is now complete"

#############

##RANDOM FORESR AND EXTREMLY RANDOM FOREST##

#argument parser
start_time = time.time()
def build_arg_parser():
	parser = argparse.ArgumentParser(description ='classify data using ensemble learning technique')
	parser.add_argument('--classifier-type', dest ='classifier_type', required='True', choices=['rf','erf'], help='Type of classifier to use; can be either "rf" or "erf"')
	return parser

if __name__ == '__main__':
		#parser the input arguments
	args = build_arg_parser().parse_args()
	classifier_type = args.classifier_type

    # Split data into training and testing datasets #
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

    #Ensemble Learning classifier

	Params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
	if classifier_type == 'rf':
		classifier = RandomForestClassifier(**Params)
	else:
		classifier = ExtraTreesClassifier(**Params)

	classifier.fit(X_train,y_train)

	y_Test_pred = classifier.predict(X_test)
	print len(y_Test_pred)
	print len(y_test)
	correct = np.array(y_test) == np.array(map(float,y_Test_pred))
	print correct
	erf_accuracy = 100.0*correct.sum() / len(X_test)
	print "the ER and ERF accuracy is ", erf_accuracy
stop_time = time.time()


#############
print "Start KNN"
###KNN###
knn_classifier = neighbors.KNeighborsClassifier()
start_time = time.time()

knn_classifier.fit(X, y)

knn_score = knn_classifier.score(X, y)
stop_time = time.time()
knn_runtime = stop_time-start_time
print "knn score is ", knn_score
print "knn runtime is ", knn_runtime
print "End KNN"
