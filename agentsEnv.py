import numpy as np 
import AnalyticGeometryFunctions as ag

class SheepPositionReset():
    def __init__(self, initSheepPosition, initSheepPositionNoise, checkBoundaryAndAdjust):
        self.initSheepPosition = initSheepPosition
        self.initSheepPositionNoise = initSheepPositionNoise
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self):
        startSheepPosition = self.initSheepPosition + np.random.uniform(-self.initSheepPositionNoise, self.initSheepPositionNoise)
        checkedPosition, toWallDistance = self.checkBoundaryAndAdjust(startSheepPosition)
        return checkedPosition

class WolfPositionReset():
    def __init__(self, initWolfPosition, initWolfPositionNoise, checkBoundaryAndAdjust):
        self.initWolfPosition = initWolfPosition
        self.initWolfPositionNoise = initWolfPositionNoise
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self):
        startWolfPosition = self.initWolfPosition + np.random.uniform(-self.initWolfPositionNoise, self.initWolfPositionNoise)
        checkedPosition, toWallDistance = self.checkBoundaryAndAdjust(startWolfPosition)
        return checkedPosition

class SheepPositionTransition():
    def __init__(self, numOneAgentState, positionIndex, checkBoundaryAndAdjust):
        self.numOneAgentState = numOneAgentState
        self.positionIndex = positionIndex
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self, oldAllAgentState, sheepId, sheepAction):
        oldSheepState = oldAllAgentState[self.numOneAgentState * sheepId : self.numOneAgentState * (sheepId + 1)]
        oldSheepPosition = oldSheepState[min(self.positionIndex) : max(self.positionIndex) + 1] 
        newSheepVelocity = sheepAction
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
        newWolfPosition = oldWolfPosition
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


