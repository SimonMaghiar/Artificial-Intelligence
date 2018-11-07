import numpy as np
import random

def playRandomGame():
	sticks = 20
	current_player = 0
	while sticks > 0:
		choice = random.randint(1,min(sticks,3))
		print (sticks, ": Player", current_player, "takes", choice )
		sticks -= choice
		current_player = 1 - current_player  # switch player 

playRandomGame()

################################################################

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units = 32, input_shape = (1,), activation='sigmoid'))
model.add(Dense(units = 32, activation='sigmoid'))
model.add(Dense(units = 32, activation='sigmoid'))
model.add(Dense(units = 1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

################################################################

def bestChoice(sticks):
	a = model.predict(np.array([[sticks-1],[sticks-2],[sticks-3]]))
	return np.argmax(a) + 1

def printChances():
	for i in range (1,21):
		print (i, ":", bestChoice(i), model.predict(np.array([i]))[0])

#printChances()

def playGame():
	sticks = 20
	current_player = 0
	while sticks > 0:
		choice = bestChoice(sticks)
		print (sticks, ": Player", current_player, "takes", choice )
		sticks -= choice
		current_player = 1 - current_player  # switch player 
#playGame()

def generatePredictedGame():
	sticks = 20
	current_player = 0
	data = []
	while sticks > 0:
		data.append(sticks)
		choice = bestChoice(sticks)
		#print (sticks, ": Player", current_player, "takes", choice )
		sticks -= choice
		current_player = 1 - current_player  # switch player 

	result = [None] * len(data)
	won_or_lost = 0;   # lost
	for i in reversed(range(len(result))):
		won_or_lost = 1 - won_or_lost
		result[i] = won_or_lost

	return np.array(data), np.array(result)

################################################################

datasetX, datasetY = generatePredictedGame()

for i in range(200):
    if i%10 == 0:
        datasetX, datasetY = generatePredictedGame()

    for i in range(50):
        game, result = generatePredictedGame()
        datasetX = np.append(datasetX,game,axis=0)
        datasetY = np.append(datasetY,result,axis=0)
    
    model.fit(datasetX, datasetY, epochs = 10, validation_split=0.3 )
    
    for j in range(1,21):
    	print(j, " : ", bestChoice(j), model.predict(np.array([j]))[0])

    print ( generatePredictedGame() )


