import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import functools as ft
import env
import reward
import dataSave 
import GenerateModel as gm
import itertools as it
import math
import pygame as pg

def approximatePolicy(stateBatch, model):
    graph = model.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    actionSample_ = graph.get_tensor_by_name('outputs/actionSample_:0')
    actionBatch = model.run(actionSample_, feed_dict = {state_ : stateBatch})
    return actionBatch

class SampleTrajectory():
    def __init__(self, episilon, actionLow, actionHigh, maxTimeStep, transitionFunction, isTerminal, reset):
        self.episilon = episilon
        self.actionLow = actionLow
        self.actionHigh = actionHigh
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, policy, episodeIndex): 
        oldState = self.reset()
        trajectory = []
        for time in range(self.maxTimeStep): 
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = policy(oldStateBatch) 
            action = actionBatch[0]
            explore = np.random.uniform(0, 1)
            expolreStand = self.episilon * (0.999 ** episodeIndex)
            if explore < expolreStand:
                action = np.random.uniform(self.actionLow, self.actionHigh)
            # actionBatch shape: batch * action Dimension; only need action Dimention
            newState = self.transitionFunction(oldState, action)
            trajectory.append((np.concatenate(oldState), action))
            
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
    def __init__(self, summaryWriter):
        self.summaryWriter = summaryWriter
    def __call__(self, episode, episodeIndex, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.array(stateEpisode).reshape(numBatch, -1), np.array(actionEpisode).reshape(numBatch, -1)
        mergedAccumulatedRewardsEpisode = np.concatenate(normalizedAccumulatedRewardsEpisode)
        
        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        accumulatedRewards_ = graph.get_tensor_by_name('inputs/accumulatedRewards_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        lossSummary = graph.get_tensor_by_name('outputs/Loss:0')
        loss, summary, trainOpt = model.run([loss_, lossSummary, trainOpt_], feed_dict={state_: stateBatch,
                                                                                        action_: actionBatch,
                                                                                        accumulatedRewards_: mergedAccumulatedRewardsEpisode
                                                                                        })
        self.summaryWriter.add_summary(summary, episodeIndex)
        self.summaryWriter.flush()
        return loss, model

class PolicyGradient():
    def __init__(self, numAgent, numTrajectory, maxEpisode, render):
        self.numAgent = numAgent
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
        self.render = render
    def __call__(self, model, approximatePolicy, sampleTrajectoryFunctions, accumulateRewardsFunctions, train):
        for episodeIndex in range(self.maxEpisode):
            policy = lambda state: approximatePolicy(state, model)
            conditions = np.random.choice(self.numAgent - 1, self.numTrajectory) 
            episode = [sampleTrajectoryFunctions[condition](policy, episodeIndex) for condition in conditions]
            normalizedAccumulatedRewardsEpisode = [normalize(accumulateRewardsFunctions[condition](trajectory)) for condition, trajectory in zip(conditions, episode)]
            loss, model = train(episode, episodeIndex, normalizedAccumulatedRewardsEpisode, model)
            #print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
            #print(loss)
            if np.mean([len(episode[index]) for index in range(self.numTrajectory)]) > 250:
                for index in range(10, 15):
                    for timeStep in episode[index]:
                        self.render(timeStep[0].reshape(2,2))
                break
        return model

def main():
#    tf.set_random_seed(123)
#    np.random.seed(123)
    numAgent = 2
    xBoundary = [0, 364]
    yBoundary = [0, 364]
    initCurrPosition = np.array([[180, 180], [90, 90]])
    wolfVelocityValue = 3
    chasingSubtleties = [50]
    minDistance = 10

    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255,255,255]
    circleColorList = [[50,255,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50]]
    circleSize = 10
    saveImage = False
    saveImageFile = 'image'
    renderOn = True
    
    numActionSpace = 2
    numStateSpace = 4
    actionLow = np.array([-10, -10])
    actionHigh = np.array([10, 10])
    actionRatio = (actionHigh - actionLow) / 2.
    episilon = 0.2


    maxTimeStep = 300

    aliveBouns = 1
    deathPenalty = 1
    rewardDecay = 1

    numTrajectory = 10 
    maxEpisode = 100000

    learningRate = 1e-3
    summaryPath = 'tensorBoard/policyGradientContinuous'

    savePath = 'data/tmpModelGaussian.ckpt'
    
    with tf.name_scope("inputs"):
        state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
        action_ = tf.placeholder(tf.float32, [None, numActionSpace], name="action_")
        accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

    with tf.name_scope("hidden"):
        fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu) 
        fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = 30, activation = tf.nn.relu) 
        actionMean_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
        actionVariance_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.softplus)

    with tf.name_scope("outputs"):        
        actionDistribution_ = tfp.distributions.MultivariateNormalDiag(actionMean_ * actionRatio, 3 * actionVariance_ + 1e-8, name = 'actionDistribution_')
        actionSample_ = tf.clip_by_value(actionDistribution_.sample(), actionLow, actionHigh, name = 'actionSample_')
        negLogProb_ = - actionDistribution_.log_prob(action_, name = 'negLogProb_')
        loss_ = tf.reduce_mean(tf.multiply(negLogProb_, accumulatedRewards_), name = 'loss_')
        tf.summary.scalar("Loss", loss_)

    with tf.name_scope("train"):
        trainOpt_ = tf.train.AdamOptimizer(learningRate, name = 'adamOpt_').minimize(loss_)

    mergedSummary = tf.summary.merge_all()
    model = tf.Session()
    model.run(tf.global_variables_initializer())    
    summaryWriter = tf.summary.FileWriter(summaryPath, graph = model.graph)
    
    checkBoundaryAndAdjustPosition = gm.CheckBoundaryAndAdjustPosition(xBoundary, yBoundary)
    physicalEngines = [gm.GenerateNextFramePosition(numAgent, wolf, chasingSubtlety, wolfVelocityValue, checkBoundaryAndAdjustPosition) 
            for wolf, chasingSubtlety in it.product(range(1, numAgent), chasingSubtleties)]

    beliefEngine = None
    render = env.Render(screen, screenColor, circleColorList, circleSize, saveImage, saveImageFile)
    transitionFunctions = [env.TransitionFunction(physicalEngine, beliefEngine, renderOn, render) for physicalEngine in physicalEngines]
    terminalFunctions = [env.IsTerminal(wolf, minDistance) for wolf in range(1, numAgent)]
    reset = env.Reset(initCurrPosition)
    sampleTrajectoryFunctions = [SampleTrajectory(episilon, actionLow, actionHigh, maxTimeStep, transitionFunction, isTerminal, reset) for transitionFunction, isTerminal in zip(transitionFunctions,
        terminalFunctions)]

    rewardFunctions = [reward.RewardFunctionTerminalPenalty(aliveBouns, deathPenalty, isTerminal) for isTerminal in terminalFunctions]
    accumulateRewardsFunctions = [AccumulateRewards(rewardDecay, rewardFunction) for rewardFunction in rewardFunctions]

    train = TrainTensorflow(summaryWriter) 

    policyGradient = PolicyGradient(numAgent, numTrajectory, maxEpisode, render)

    trainedModel = policyGradient(model, approximatePolicy, sampleTrajectoryFunctions, accumulateRewardsFunctions, train)

    saveModel = dataSave.SaveModel(savePath)
    modelSave = saveModel(model)

if __name__ == "__main__":
    main()
