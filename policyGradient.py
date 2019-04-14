import tensorflow as tf
import numpy as np
import functools as ft
import env
import reward
import dataSave 
import tensorflow_probability as tfp
import random
import agentsEnv as ag
import itertools as it
import pygame as pg

class ApproximatePolicy():
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.numActionSpace = len(self.actionSpace)
    def __call__(self, stateBatch, model):
        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionDistribution_ = graph.get_tensor_by_name('outputs/actionDistribution_:0')
        actionDistributionBatch = model.run(actionDistribution_, feed_dict = {state_ : stateBatch})
        actionIndexBatch = [np.random.choice(range(self.numActionSpace), p = actionDistribution) for actionDistribution in actionDistributionBatch]
        actionBatch = np.array([self.actionSpace[actionIndex] for actionIndex in actionIndexBatch])
        return actionBatch

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal

    def __call__(self, actor): 
        oldState , action = None, None
        oldState = self.transitionFunction(oldState, action)
        trajectory = []
        
        for time in range(self.maxTimeStep):
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = actor(oldStateBatch)

            action = actionBatch[0] 
            # actionBatch shape: batch * action Dimension; only keep action Dimention in shape
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))
            terminal = self.isTerminal(oldState)
            if terminal:
                break
            oldState = newState
        return trajectory


class AccumulateRewards():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])
        return accumulatedRewards

def normalize(accumulatedRewards):
    normalizedAccumulatedRewards = (accumulatedRewards - np.mean(accumulatedRewards)) / np.std(accumulatedRewards)
    return normalizedAccumulatedRewards

class TrainTensorflow():
    def __init__(self, actionSpace, summaryPath):
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)
        self.summaryWriter = tf.summary.FileWriter(summaryPath)
    def __call__(self, episode, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        actionIndexEpisode = np.array([list(self.actionSpace).index(list(action)) for action in actionEpisode])
        actionLabelEpisode = np.zeros([numBatch, self.numActionSpace])
        actionLabelEpisode[np.arange(numBatch), actionIndexEpisode] = 1
        stateBatch, actionLabelBatch = np.array(stateEpisode).reshape(numBatch, -1), np.array(actionLabelEpisode).reshape(numBatch, -1)
        mergedAccumulatedRewardsEpisode = np.concatenate(normalizedAccumulatedRewardsEpisode)

        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionLabel_ = graph.get_tensor_by_name('inputs/actionLabel_:0')
        accumulatedRewards_ = graph.get_tensor_by_name('inputs/accumulatedRewards_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = model.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                    actionLabel_ : actionLabelBatch,
                                                                    accumulatedRewards_ : mergedAccumulatedRewardsEpisode
                                                                    })
        self.summaryWriter.flush()
        return loss, model

class PolicyGradient():
    def __init__(self, numTrajectory, maxEpisode, render):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
        self.render = render
    def __call__(self, model, approximatePolicy, sampleTrajectoryFunctions, accumulateRewardsFunctions, train):
        numConditions = len(sampleTrajectoryFunctions)
        for episodeIndex in range(self.maxEpisode):
            policy = lambda state: approximatePolicy(state, model)
            conditions = np.random.choice(range(numConditions), self.numTrajectory) 
            episode = [sampleTrajectoryFunctions[condition](policy) for condition in conditions]
            normalizedAccumulatedRewardsEpisode = [normalize(accumulateRewardsFunctions[condition](trajectory)) for condition, trajectory in zip(conditions, episode)]
            loss, model = train(episode, normalizedAccumulatedRewardsEpisode, model)
            print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
            if episodeIndex%1 == 0:
                for index in range(10, 15):
                    for timeStep in episode[index]:
                        self.render(timeStep[0])
        return model

def main():
    actionSpace = [[10,0],[7,7],[0,10],[-7,7],[-10,0],[-7,-7],[0,-10],[7,-7]]
    numActionSpace = len(actionSpace)
    numStateSpace = 4
    summaryPath = 'tensorBoard/policyGradient'

    savePath = 'data/tmpModel.ckpt'
    learningRate = 1e-9

    with tf.name_scope("inputs"):
        state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
        actionLabel_ = tf.placeholder(tf.int32, [None, numActionSpace], name="actionLabel_")
        accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

    with tf.name_scope("hidden"):
        fullyConnected1_ = tf.layers.dense(inputs = state_, units = 40, activation = tf.nn.relu)
        fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = 40, activation = tf.nn.relu)
        fullyConnected3_ = tf.layers.dense(inputs = fullyConnected2_, units = 50, activation = tf.nn.relu)
        fullyConnected4_ = tf.layers.dense(inputs = fullyConnected3_, units = 50, activation = tf.nn.relu)
        fullyConnected5_ = tf.layers.dense(inputs = fullyConnected4_, units = 50, activation = tf.nn.relu)
        fullyConnected6_ = tf.layers.dense(inputs = fullyConnected5_, units = 50, activation = tf.nn.relu)
        fullyConnected7_ = tf.layers.dense(inputs = fullyConnected6_, units = 50, activation = tf.nn.relu)
        fullyConnected8_ = tf.layers.dense(inputs = fullyConnected7_, units = 40, activation = tf.nn.relu)
        fullyConnected9_ = tf.layers.dense(inputs = fullyConnected8_, units = 40, activation = tf.nn.relu)
        allActionActivation_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = None)

    with tf.name_scope("outputs"):
        actionDistribution_ = tf.nn.softmax(allActionActivation_, name = 'actionDistribution_')
        negLogProb_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits = allActionActivation_, labels = actionLabel_, name = 'negLogProb_')
        loss_ = tf.reduce_mean(tf.multiply(negLogProb_, accumulatedRewards_), name = 'loss_')
    tf.summary.scalar("Loss", loss_)

    with tf.name_scope("train"):
        trainOpt_ = tf.train.AdamOptimizer(learningRate, name = 'adamOpt_').minimize(loss_)

    mergedSummary = tf.summary.merge_all()
    
    model = tf.Session()
    model.run(tf.global_variables_initializer())    

    approximatePolicy = ApproximatePolicy(actionSpace)

    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    
    initSheepPosition = np.array([180, 180]) 
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([90, 90])
    initWolfPositionNoise = np.array([0, 0])
    sheepPositionAndVelocityReset = ag.SheepPositionAndVelocityReset(initSheepPosition, initSheepVelocity, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionAndVelocityReset = ag.WolfPositionAndVelocityReset(initWolfPosition, initWolfVelocity, initWolfPositionNoise, checkBoundaryAndAdjust)
    
    numOneAgentState = 2
    positionIndex = [0, 1]
    velocityIndex = [2, 3]
    sheepVelocitySpeed = 10
    sheepActionFrequency = 1
    wolfVelocitySpeed = 0
    wolfActionFrequency = 1
    sheepPositionAndVelocityTransation = ag.SheepPositionAndVelocityTransation(sheepVelocitySpeed, sheepActionFrequency, 
            numOneAgentState, positionIndex, velocityIndex, checkBoundaryAndAdjust) 
    wolfPositionAndVelocityTransation = ag.WolfPositionAndVelocityTransation(wolfVelocitySpeed, wolfActionFrequency,
            numOneAgentState, positionIndex, velocityIndex, checkBoundaryAndAdjust) 
    
    numAgent = 2
    sheepIndexOfId = 0
    wolfIndexOfId = 1
    originAgentId = list(range(numAgent))
    #fixedId for sheep
    fixedIds= list(range(0, 1))
    #unfixedId for wolf and distractors
    unfixedIds = list(range(1, numAgent))
    possibleUnfixedIds = it.permutations(unfixedIds)
    possibleAgentIds = [fixedIds + list(unfixedIds) for unfixedIds in possibleUnfixedIds]
    possibleWolfSubtleties = [50]
    conditions = it.product(possibleAgentIds, possibleWolfSubtleties)
    transitionFunctions = [env.TransitionFunction(agentIds, sheepIndexOfId, wolfIndexOfId, wolfSubtlety, 
        sheepPositionAndVelocityReset, wolfPositionAndVelocityReset, sheepPositionAndVelocityTransation, wolfPositionAndVelocityTransation) 
        for agentIds, wolfSubtlety in conditions]
    
    minDistance = 15
    isTerminals = [env.IsTerminal(agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex, 
        minDistance) for agentIds in possibleAgentIds]
    
    maxTimeStep = 60 
    sampleTrajectoryFunctions = [SampleTrajectory(maxTimeStep, transitionFunction, isTerminal) for transitionFunction, isTerminal in zip(transitionFunctions, isTerminals)]
    
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
    rewardFunctions = [reward.RewardFunctionTerminalPenalty(agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex,
        aliveBouns, deathPenalty, isTerminal) for agentIds, isTerminal in zip(possibleAgentIds, isTerminals)] 

    #rewardFunction = reward.CartpoleRewardFunction(aliveBouns)
    accumulateRewardsFunctions = [AccumulateRewards(rewardDecay, rewardFunction) for rewardFunction in rewardFunctions]

    train = TrainTensorflow(actionSpace, summaryPath) 

    numTrajectory = 150
    maxEpisode = 100000
    policyGradient = PolicyGradient(numTrajectory, maxEpisode, render)

    trainedModel = policyGradient(model, approximatePolicy, sampleTrajectoryFunctions, accumulateRewardsFunctions, train)

    saveModel = dataSave.SaveModel(savePath)
    modelSave = saveModel(model)


if __name__ == "__main__":
    main()
