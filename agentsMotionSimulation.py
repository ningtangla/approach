import numpy as np 
import AnalyticGeometryFunctions as ag

class SheepPositionReset():
    def __init__(self, initSheepPosition, initSheepPositionNoise):
        self.initSheepPosition = initSheepPosition
        self.initSheepPositionNoiseLow, self.initSheepPositionNoiseHigh = initSheepPositionNoise
    def __call__(self):
        noise = [np.random.uniform(self.initSheepPositionNoiseLow, self.initSheepPositionNoiseHigh) * np.random.choice([-1, 1]) for dim in range(len(self.initSheepPosition))]
        startSheepPosition = self.initSheepPosition + np.array(noise)
        return startSheepPosition

class WolfPositionReset():
    def __init__(self, initWolfPosition, initWolfPositionNoise):
        self.initWolfPosition = initWolfPosition
        self.initWolfPositionNoiseLow, self.initWolfPositionNoiseHigh = initWolfPositionNoise
    def __call__(self):
        noise = [np.random.uniform(self.initWolfPositionNoiseLow, self.initWolfPositionNoiseHigh) * np.random.choice([-1, 1]) for dim in range(len(self.initWolfPosition))]
        startWolfPosition = self.initWolfPosition + np.array(noise)
        return startWolfPosition

class SheepPositionTransition():
    def __init__(self, numOneAgentState, positionIndex, checkBoundaryAndAdjust):
        self.numOneAgentState = numOneAgentState
        self.positionIndex = positionIndex
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self, oldAllAgentState, sheepId, sheepAction):
        oldSheepState = oldAllAgentState[self.numOneAgentState * sheepId : self.numOneAgentState * (sheepId + 1)]
        oldSheepPosition = oldSheepState[min(self.positionIndex) : max(self.positionIndex) + 1] 
        newSheepVelocity = np.array(sheepAction)
        newSheepPosition = oldSheepPosition + newSheepVelocity
        checkedPosition, toWallDistance = self.checkBoundaryAndAdjust(newSheepPosition)
        return checkedPosition

class WolfPositionTransition():
    def __init__(self, numOneAgentState, positionIndex, checkBoundaryAndAdjust):
        self.numOneAgentState = numOneAgentState
        self.positionIndex = positionIndex
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self, oldAllAgentState, wolfId):
        oldWolfState = oldAllAgentState[self.numOneAgentState * wolfId : self.numOneAgentState * (wolfId + 1)]
        oldWolfPosition = oldWolfState[min(self.positionIndex) : max(self.positionIndex) + 1] 
        newWolfPosition = np.array(oldWolfPosition)
        checkedPosition, toWallDistance = self.checkBoundaryAndAdjust(newWolfPosition)
        return checkedPosition

class CheckBoundaryAndAdjust():
    def __init__(self, xBoundary, yBoundary):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
    def __call__(self, position):
        if position[0] >= self.xMax:
            position[0] = 2 * self.xMax - position[0]
        if position[0] <= self.xMin:
            position[0] = 2 * self.xMin - position[0]
        if position[1] >= self.yMax:
            position[1] = 2 * self.yMax - position[1]
        if position[1] <= self.yMin:
            position[1] = 2 * self.yMin - position[1]

        toWallDistance = np.concatenate([position[0] - self.xBoundary, position[1] - self.yBoundary, self.xBoundary, self.yBoundary])
        return position, toWallDistance    


