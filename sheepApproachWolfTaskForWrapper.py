
import numpy as np
import pygame as pg

import agentsMotionSimulation as ag
import env
import reward

def main():

    actionSpace = [[10,0],[7,7],[0,10],[-7,7],[-10,0],[-7,-7],[0,-10],[7,-7]]
    numActionSpace = len(actionSpace)
    numStateSpace = 4
   
    initSheepPosition = np.array([180, 180]) 
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([60, 120])
    initWolfPositionNoise = np.array([0, 60])
    sheepPositionReset = ag.SheepPositionReset(initSheepPosition, initSheepPositionNoise)
    wolfPositionReset = ag.WolfPositionReset(initWolfPosition, initWolfPositionNoise)
    
    numOneAgentState = 2
    positionIndex = [0, 1]
    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary) 
    sheepPositionTransition = ag.SheepPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust) 
    wolfPositionTransition = ag.WolfPositionTransition(numOneAgentState, positionIndex, checkBoundaryAndAdjust) 
    
    numAgent = 2
    sheepId = 0
    wolfId = 1
    transitionFunction = env.TransitionFunction(sheepId, wolfId, sheepPositionReset, wolfPositionReset, sheepPositionTransition, wolfPositionTransition)
    minDistance = 15
    isTerminal = env.IsTerminal(sheepId, wolfId, numOneAgentState, positionIndex, minDistance) 
     
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255,255,255]
    circleColorList = [[50,255,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50]]
    circleSize = 8
    saveImage = False
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageFile)

    aliveBouns = -1
    deathPenalty = 20
    rewardDecay = 0.99
    rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, wolfId, numOneAgentState, positionIndex, aliveBouns, deathPenalty, isTerminal) 
    accumulateReward = AccumulateReward(rewardDecay, rewardFunction)
