import numpy as np
import scipy as sp
import math
import AnalyticGeometryFunctions as ag
import pandas as pd

class GenerateNextFramePosition():
    def __init__(self, numAgent, wolf, chasingSubtletyForWolfVelocityDirection, wolfVelocityValue, checkBoundaryAndAdjustPosition): 
        self.numAgent = numAgent 
        self.wolf = wolf
        self.chasingSubtletyForWolfVelocityDirection = chasingSubtletyForWolfVelocityDirection
        self.wolfVelocityValue = wolfVelocityValue
        self.checkBoundaryAndAdjustPosition = checkBoundaryAndAdjustPosition
        self.sheep = 0
    
    def __call__(self, currPosition, sheepAction):

        wolfCurrPosition, sheepCurrPosition = currPosition.loc[self.wolf], currPosition.loc[self.sheep]
        heatSeekingDirectionCartesian = sheepCurrPosition - wolfCurrPosition
        heatSeekingDirectionPolar = ag.transiteDirectionCartesianToPolar(heatSeekingDirectionCartesian)
        wolfVelocityDirectionPolar = np.random.vonmises(heatSeekingDirectionPolar, self.chasingSubtletyForWolfVelocityDirection) 

        agentsNumber = len(currPosition)
        agentsVelocityDirectionPolar = [wolfVelocityDirectionPolar if agent == self.wolf \
                else np.random.uniform(-math.pi, math.pi) for agent in range(agentsNumber)]

        agentsVelocityDirectionCartesian = [ag.transiteDirectionPolarToCartesian(agentVelocityDirectionPolar) \
                for agentVelocityDirectionPolar in agentsVelocityDirectionPolar] 
        agentsVelocityValue = [self.wolfVelocityValue] * (self.numAgent)  
        agentsVelocity = np.array(agentsVelocityDirectionCartesian) * np.array(agentsVelocityValue)
        agentsVelocity[0] = sheepAction
        nextPositionBeforeBoundaryCheck = currPosition.values + agentsVelocity
        nextPosition = pd.DataFrame(np.array([self.checkBoundaryAndAdjustPosition(nextPositionBeforeBoundaryCheck[agent]) \
                for agent in range(agentsNumber)]), index = currPosition.index, columns = currPosition.columns)
        return nextPosition

class CheckBoundaryAndAdjustPosition():
    def __init__(self, xBoundary, yBoundary):
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
        return position        
