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
        #entropy_ = graph.get_tensor_by_name('outputs/actionEntropy_:0')
        #actionEntropyBatch = model.run(entropy_, feed_dict = {state_ : stateBatch})
        #__import__('ipdb').set_trace()
        actionIndexBatch = [np.random.choice(range(self.numActionSpace), p = actionDistribution) for actionDistribution in actionDistributionBatch]
        actionBatch = np.array([self.actionSpace[actionIndex] for actionIndex in actionIndexBatch])
        return actionBatch

def approximateValue(stateBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
    valueBatch = criticModel.run(value_, feed_dict = {state_ : stateBatch})
    return valueBatch

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay):
        self.criticWriter = criticWriter
        self.decay = decay
    def __call__(self, state, action, nextState, reward, criticModel):
        
        stateBatch, actionBatch, nextStateBatch = state.reshape(1, -1), action.reshape(1, -1), nextState.reshape(1, -1)
        rewardBatch = reward.reshape(1, -1) 

        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
        nextStateValueBatch = criticModel.run(value_, feed_dict = {state_ : nextStateBatch})
        
        valueTargetBatch = rewardBatch + self.decay * nextStateValueBatch

        state_ = graph.get_tensor_by_name('inputs/state_:0')
        valueTarget_ = graph.get_tensor_by_name('inputs/valueTarget_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          valueTarget_ : valueTargetBatch
                                                                          })
        self.criticWriter.flush()
        return loss, criticModel

class EstimateAdvantageBootstrap():
    def __init__(self, decay):
        self.decay = decay
    def __call__(self, state, action, nextState, reward, critic):
        
        stateBatch, actionBatch, nextStateBatch = state.reshape(1, -1), action.reshape(1, -1), nextState.reshape(1, -1)
        rewardBatch = reward.reshape(1, -1) 
        advantageBatch = rewardBatch + self.decay * critic(nextStateBatch) - critic(stateBatch)
        advantages = np.concatenate(advantageBatch)
        return advantages

class TrainActorTensorflow():
    def __init__(self, actionSpace, actorWriter):
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)
        self.actorWriter = actorWriter
    def __call__(self, state, action, advantages, actorModel):
        actionIndex = list(self.actionSpace).index(list(action))
        actionLabel = np.zeros([self.numActionSpace])
        actionLabel[actionIndex] = 1
        stateBatch, actionLabelBatch = np.array(state).reshape(1, -1), np.array(actionLabel).reshape(1, -1)
        
        graph = actorModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionLabel_ = graph.get_tensor_by_name('inputs/actionLabel_:0')
        advantages_ = graph.get_tensor_by_name('inputs/advantages_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = actorModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                         actionLabel_ : actionLabelBatch,
                                                                         advantages_ : advantages            
                                                                         })
        return loss, actorModel

class OnlineAdvantageActorCritic():
    def __init__(self, maxEpisode, maxTimeStep, transitionFunctions, isTerminals, rewardFunctions, render):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.transitionFunctions = transitionFunctions
        self.isTerminals = isTerminals
        self.rewardFunctions = rewardFunctions
        self.numCondition = len(transitionFunctions)
        self.render = render
    def __call__(self, actorModel, criticModel, approximatePolicy, trainCritic, approximateValue, estimateAdvantage, trainActor):
        for episodeIndex in range(self.maxEpisode):
            conditionIndex = np.random.choice(range(self.numCondition))
            self.transitionFunction = self.transitionFunctions[conditionIndex]
            self.isTerminal = self.isTerminals[conditionIndex]
            self.rewardFunction = self.rewardFunctions[conditionIndex]
            
            oldState, action = None, None
            oldState = self.transitionFunction(oldState, action)
            for timeStepIndex in range(self.maxTimeStep):
                actor = lambda state: approximatePolicy(state, actorModel)
                actionBatch = actor(oldState.reshape(1, -1))
                action = actionBatch[0]
                newState = self.transitionFunction(oldState, action)
                reward = self.rewardFunction(oldState, action)
                valueLoss, criticModel = trainCritic(oldState, action, newState, reward, criticModel)
                critic = lambda state: approximateValue(state, criticModel)
                advantage = estimateAdvantage(oldState, action, newState, reward, critic)
                policyLoss, actorModel = trainActor(oldState, action, advantage, actorModel)
                if self.isTerminal(oldState):
                    break
                oldState = newState
                if episodeIndex % 1 == 0:
                    self.render(oldState)
            print(timeStepIndex)
        return actorModel, criticModel

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)

    actionSpace = [[10,0],[8.5, 5],[7,7],[5,8.5],[0,10],[-5, 8.5],[-7,7],[-8.5,5],[-10,0],[-8.5,-5],[-7,-7],[-5,-8.5],[0,-10],[5,-8.5],[7,-7],[8.5,-5]]
    numActionSpace = len(actionSpace)
    numStateSpace = 24

    numActorFC1Unit = 60
    numActorFC2Unit = 60
    numActorFC3Unit = 60
    numActorFC4Unit = 60
    numCriticFC1Unit = 100
    numCriticFC2Unit = 100
    numCriticFC3Unit = 100
    numCriticFC4Unit = 100
    learningRateActor = 1e-4
    learningRateCritic = 3e-4
 
    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            actionLabel_ = tf.placeholder(tf.int32, [None, numActionSpace], name="actionLabel_")
            advantages_ = tf.placeholder(tf.float32, [None, ], name="advantages_")

        with tf.name_scope("hidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.01)
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = numActorFC1Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            fullyConnected3_ = tf.layers.dense(inputs = fullyConnected2_, units = numActorFC2Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            allActionActivation_ = tf.layers.dense(inputs = fullyConnected3_, units = numActionSpace, activation = None, kernel_initializer = initWeight, bias_initializer = initBias )

        with tf.name_scope("outputs"):
            actionDistribution_ = tf.nn.softmax(allActionActivation_, name = 'actionDistribution_')
            actionEntropy_ = tf.multiply(tfp.distributions.Categorical(probs = actionDistribution_).entropy(), 1, name = 'actionEntropy_') 
            negLogProb_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits = allActionActivation_, labels = actionLabel_, name = 'negLogProb_')
            loss_ = tf.reduce_mean(tf.multiply(negLogProb_, advantages_) - actionEntropy_, name = 'loss_') 
            actorLossSummary = tf.summary.scalar("ActorLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateActor, name = 'adamOpt_').minimize(loss_)

        actorInit = tf.global_variables_initializer()
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorWriter = tf.summary.FileWriter('tensorBoard/actorOfflineA2C', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

        with tf.name_scope("hidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = numActorFC1Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            fullyConnected3_ = tf.layers.dense(inputs = fullyConnected1_, units = numActorFC3Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            fullyConnected4_ = tf.layers.dense(inputs = fullyConnected1_, units = numActorFC4Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )

        with tf.name_scope("outputs"):        
            value_ = tf.layers.dense(inputs = fullyConnected4_, units = 1, activation = None, name = 'value_', kernel_initializer = initWeight, bias_initializer = initBias )
            diff_ = tf.subtract(valueTarget_, value_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
        criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/criticOfflineA2C', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)    
     
    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    
    initSheepPosition = np.array([180, 180]) 
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([119, 119])
    initWolfPositionNoise = np.array([59, 59])
    sheepPositionAndVelocityReset = ag.SheepPositionAndVelocityReset(initSheepPosition, initSheepVelocity, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionAndVelocityReset = ag.WolfPositionAndVelocityReset(initWolfPosition, initWolfVelocity, initWolfPositionNoise, checkBoundaryAndAdjust)
    
    numOneAgentState = 12
    positionIndex = [0, 1]
    velocityIndex = [2, 3]
    sheepVelocitySpeed = 10
    sheepActionFrequency = 1
    wolfVelocitySpeed = 6
    wolfActionFrequency = 12
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
    
    minDistance = 10
    isTerminals = [env.IsTerminal(agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex, 
        minDistance) for agentIds in possibleAgentIds]
     
    screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
    screenColor = [255,255,255]
    circleColorList = [[50,255,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50],[50,50,50]]
    circleSize = 8
    saveImage = False
    saveImageFile = 'image'
    render = env.Render(numAgent, numOneAgentState, positionIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageFile)

    aliveBouns = 1
    deathPenalty = -20
    rewardDecay = 0.99
    rewardFunctions = [reward.RewardFunctionTerminalPenalty(agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex,
        aliveBouns, deathPenalty, isTerminal) for agentIds, isTerminal in zip(possibleAgentIds, isTerminals)] 
    
    approximatePolicy = ApproximatePolicy(actionSpace)
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay)
    estimateAdvantage = EstimateAdvantageBootstrap(rewardDecay)
    
    trainActor = TrainActorTensorflow(actionSpace, actorWriter) 

    maxTimeStep = 200
    maxEpisode = 100000
    actorCritic = OnlineAdvantageActorCritic(maxEpisode, maxTimeStep, transitionFunctions, isTerminals, rewardFunctions, render)

    trainedActorModel, trainedCriticModel = actorCritic(actorModel, criticModel, approximatePolicy, trainCritic,
            approximateValue, estimateAdvantage, trainActor)

    with actorModel.as_default():
        actorSaver.save(trainedActorModel, savePathActor)
    with criticModel.as_default():
        criticSaver.save(trainedCriticModel, savePathCritic)

if __name__ == "__main__":
    main()

