import numpy as np
from sklearn.preprocessing import normalize
import os
import copy

## forward Selection funtion
def forwardSelection(data, maxSet, maxAccuracy):
	row, column = data.shape
	bestAccuracy = maxAccuracy
	currentSet = []
	for i in range(1, column):
		addFeature = 0
		currAccuracy = 0
		for j in maxSet:
			if not j in currentSet:
				tempSet = currentSet + [j]
				accuracy = oneoutValidator(data, tempSet, row)
				#print("Using feature(s) " + str(currentSet) 
				#			+ ", accuracy is " + str(currAccuracy) + "%\n")
				if accuracy > currAccuracy:
					currAccuracy = accuracy
					addFeature = j
		
		currentSet.append(addFeature)
		print("Best accuracy are feature(s) " + str(currentSet) + ", accuracy is " + "{0:.1f}".format(currAccuracy) + "%\n")
		
		if currAccuracy > bestAccuracy:
			bestAccuracy = currAccuracy
			bestSet = copy.deepcopy(currentSet)
		elif currAccuracy < bestAccuracy:
			print("(Warning, Accuracy has decreased! " 
						+ "Continuing search in case of local maxima)\n")
			##print("Using feature(s) " + str(bestSet) + " accuracy is " + str(accuracy) + "%\n")
	return bestSet, bestAccuracy

## backward elimination function
def backwardElimination(data, maxSet, maxAccuracy):
	row, column = data.shape
	bestAccuracy = maxAccuracy
	currentSet = [1,2,3,4,5,6,7,8,9,10]
	for i in range(1, column):
		addFeature = 0
		currAccuracy = 0
		for j in maxSet:
			if j in currentSet:
				tempSet = currentSet + []
				tempSet.remove(j)
				accuracy = oneoutValidator(data, tempSet, row)
				#print("Using feature(s) " + str(currentSet) 
				#			+ ", accuracy is " + str(currAccuracy) + "%\n")
				if accuracy > currAccuracy:
					currAccuracy = accuracy
					addFeature = j
		currentSet.remove(addFeature)
		print("Best accuracy are feature(s) " + str(currentSet) + ", accuracy is " + "{0:.1f}".format(currAccuracy) + "%\n")
		if currAccuracy > bestAccuracy:
			bestAccuracy = currAccuracy
			bestSet = copy.deepcopy(currentSet)
		elif currAccuracy < bestAccuracy:
			print("(Warning, Accuracy has decreased! " 
						+ "Continuing search in case of local maxima)\n")
			##print("Using feature(s) " + str(bestSet) + " accuracy is " + str(accuracy) + "%\n")
	return bestSet, bestAccuracy

## classifier function
def nnClassifier(data, datapoint, subFeature, numInstances):
    nearestneighbor = 0
    shortestDist = float('inf')
    for i in range(numInstances):
        if (datapoint != i): 
            distance = 0
            for j in subFeature:
                distance = distance + pow(
                    (data[i][j] - data[datapoint][j]),
                    2)
            distance = pow(distance, 0.5)
            if distance < shortestDist:
                nearestneighbor = i
                shortestDist = distance
    return nearestneighbor

## evaluation function
def oneoutValidator(data, subFeatures, numInstances):
	row, column = data.shape
	correctInstances = 0
	for i in range(numInstances):
		leaveOne = i
		neighbor = nnClassifier(data, leaveOne, subFeatures, numInstances)
		if data[neighbor][0] == data[leaveOne][0]:
			correctInstances = correctInstances + 1
	accuracy = (correctInstances / (numInstances-1)) * 100
	print("Using feature(s) " + str(subFeatures) + ", accuracy is " + "{0:.1f}".format(accuracy) + "%")
	return accuracy




## Main
#if __name__ == "__main__":
print("Welcome to Duke Pham Feature Selection Algorithm.")
filename = input("Type in the name of the file to test: ")
if not os.path.isfile(filename):
		print("Error: File does not exist")
		exit(1)

## load in txt file and convert to 2D array
array = np.loadtxt(filename)

	
row, column = array.shape
## set the number of features and instances
features = column - 1
instances = row

fullFeature = list(range(1, column))
normalized = copy.deepcopy(array)
normalized = normalize(normalized[:, fullFeature], axis = 0)
array = np.concatenate((array[:, [0]], normalized), axis=1)


accuracy = oneoutValidator(array, fullFeature, instances)

algoChoice = input("Type the number of the algorithm you want to run.\n" +
									 "1: Forward Selection\n" + "2: Backward Elimination\n" +
									 "3: Duke's Special Algorithm.\n")
if algoChoice != "1" and algoChoice != "2" and algoChoice != "3":
		print("Error: Incorrect choice of algorithm entered")
		exit(1)

print("This dataset has " + str(features) + " features" +
			" (not including the class attributes)" + ", with " +
			str(instances) + " instances.\n")

print("Running the nearest neighbor with no features (default rate)," +
			" using 'leaving-one-out' evaluation," + " I get an accuracy of " +
			str(accuracy) + "%")
print("\n")
print("Beginning Search\n")

if algoChoice == "1":
		print("forward Selection")
		bestSet, bestAccuracy = forwardSelection(array, fullFeature, accuracy)
elif algoChoice == "2":
		print("backward Elimination")
		bestSet, bestAccuracy = backwardElimination(array, fullFeature,
																								accuracy)
	

print("Finished Search!! The best feature subset is " + str(bestSet) +
					", which has an accuracy of " + "{0:.1f}".format(bestAccuracy) + "%")
