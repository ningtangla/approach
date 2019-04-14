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

def approximatePolicyEvaluation(stateBatch, actorModel):
    graph = actorModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    evaAction_ = graph.get_tensor_by_name('outputs/evaAction_:0')
    evaActionBatch = actorModel.run(evaAction_, feed_dict = {state_ : stateBatch})
    return evaActionBatch

def approximatePolicyTarget(stateBatch, actorModel):
    graph = actorModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    tarAction_ = graph.get_tensor_by_name('outputs/tarAction_:0')
    tarActionBatch = actorModel.run(tarAction_, feed_dict = {state_ : stateBatch})
    return tarActionBatch

def approximateQTarget(stateBatch, actionBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    action_ = graph.get_tensor_by_name('inputs/action_:0')
    tarQ_ = graph.get_tensor_by_name('outputs/tarQ_:0')
    tarQBatch = criticModel.run(tarQ_, feed_dict = {state_ : stateBatch,
                                                   action_ : actionBatch
                                                   })
    return tarQBatch

def gradientPartialActionFromQEvaluation(stateBatch, actionBatch, criticModel):
    criticGraph = criticModel.graph
    state_ = criticGraph.get_tensor_by_name('inputs/state_:0')
    action_ = criticGraph.get_tensor_by_name('inputs/action_:0')
    gradientQPartialAction_ = criticGraph.get_tensor_by_name('outputs/gradientQPartialAction_/evaluationHidden/MatMul_1_grad/MatMul:0')
    gradientQPartialAction = criticModel.run([gradientQPartialAction_], feed_dict = {state_ : stateBatch,
                                                                                     action_ : actionBatch,
                                                                                     })
    return gradientQPartialAction

class AddActionNoise():
    def __init__(self, actionNoise, noiseDecay, actionLow, actionHigh):
        self.actionNoise = actionNoise
        self.noiseDecay = noiseDecay
        self.actionLow, self.actionHigh = actionLow, actionHigh
        
    def __call__(self, actionPerfect, episodeIndex):
        noiseRange = self.actionNoise * (self.noiseDecay ** episodeIndex)
        noisyAction = actionPerfect + np.random.uniform(-noiseRange, noiseRange)
        action = np.clip(noisyAction, self.actionLow, self.actionHigh)
        return action

class Memory():
    def __init__(self, memoryCapacity):
        self.memoryCapacity = memoryCapacity
    def __call__(self, replayBuffer, timeStep):
        replayBuffer.append(timeStep)
        if len(replayBuffer) > self.memoryCapacity:
            numDelete = len(replayBuffer) - self.memoryCapacity
            del replayBuffer[numDelete : ]
        return replayBuffer

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, numMiniBatch):
        self.criticWriter = criticWriter
        self.decay = decay
        self.numBatch = numMiniBatch

    def __call__(self, states, actions, nextStates, rewards, tarActor, tarCritic, criticModel):
        
        stateBatch, actionBatch, nextStateBatch = np.array(states).reshape(self.numBatch, -1), np.array(actions).reshape(self.numBatch, -1), np.array(nextStates).reshape(self.numBatch, -1),
        rewardBatch = np.array(rewards).reshape(self.numBatch, -1)
        
        nextTargetActionBatch = tarActor(nextStateBatch)

        nextTargetQBatch = tarCritic(nextStateBatch, nextTargetActionBatch)
        
        QTargetBatch = rewardBatch + self.decay * nextTargetQBatch
        
        criticGraph = criticModel.graph
        state_ = criticGraph.get_tensor_by_name('inputs/state_:0')
        action_ = criticGraph.get_tensor_by_name('inputs/action_:0') 
        QTarget_ = criticGraph.get_tensor_by_name('inputs/QTarget_:0')
        loss_ = criticGraph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = criticGraph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          action_ : actionBatch,
                                                                          QTarget_ : QTargetBatch
                                                                          })
        
        numParams_ = criticGraph.get_tensor_by_name('outputs/numParams_:0')
        numParams = criticModel.run(numParams_)
        updateTargetParameter_ = [criticGraph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        criticModel.run(updateTargetParameter_)
        
        self.criticWriter.flush()
        return loss, criticModel

class TrainActorTensorflow():
    def __init__(self, actorWriter, numMiniBatch):
        self.actorWriter = actorWriter
        self.numBatch = numMiniBatch
    def __call__(self, states, evaActor, gradientEvaCritic, actorModel):

        stateBatch = np.array(states).reshape(self.numBatch, -1)
        evaActorActionBatch = evaActor(stateBatch)
        
        gradientQPartialAction = gradientEvaCritic(stateBatch, evaActorActionBatch)

        actorGraph = actorModel.graph
        state_ = actorGraph.get_tensor_by_name('inputs/state_:0')
        gradientQPartialAction_ = actorGraph.get_tensor_by_name('inputs/gradientQPartialAction_:0')
        gradientQPartialActorParameter_ = actorGraph.get_tensor_by_name('outputs/gradientQPartialActorParameter_/evaluationHidden/dense/MatMul_grad/MatMul:0')
        trainOpt_ = actorGraph.get_operation_by_name('train/adamOpt_')
        gradientQPartialActorParameter_, trainOpt = actorModel.run([gradientQPartialActorParameter_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                                                                              gradientQPartialAction_ : gradientQPartialAction[0]  
                                                                                                                              })
        numParams_ = actorGraph.get_tensor_by_name('outputs/numParams_:0')
        numParams = actorModel.run(numParams_)
        updateTargetParameter_ = [actorGraph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        actorModel.run(updateTargetParameter_)
        self.actorWriter.flush()
        return gradientQPartialActorParameter_, actorModel

class OnlineDeepDeterministicPolicyGradient():
    def __init__(self, maxEpisode, maxTimeStep, numMiniBatch, transitionFunctions, isTerminals, addActionNoise, rewardFunctions, render):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.numMiniBatch = numMiniBatch
        self.transitionFunctions = transitionFunctions
        self.isTerminals = isTerminals
        self.addActionNoise = addActionNoise
        self.rewardFunctions = rewardFunctions
        self.render = render
        self.numCondition = len(transitionFunctions)
    def __call__(self, actorModel, criticModel, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget, gradientPartialActionFromQEvaluation,
            memory, trainCritic, trainActor):
        replayBuffer = []
        for episodeIndex in range(self.maxEpisode):
            conditionIndex = np.random.choice(range(self.numCondition))
            self.transitionFunction = self.transitionFunctions[conditionIndex]
            self.isTerminal = self.isTerminals[conditionIndex]
            self.rewardFunction = self.rewardFunctions[conditionIndex]
            
            oldState, action = None, None
            oldState = self.transitionFunction(oldState, action)
            for timeStepIndex in range(self.maxTimeStep):
                evaActor = lambda state: approximatePolicyEvaluation(state, actorModel)
                actionBatch = evaActor(oldState.reshape(1, -1))
                actionPerfect = actionBatch[0]
                action = self.addActionNoise(actionPerfect, episodeIndex)
                newState = self.transitionFunction(oldState, action)
                reward = self.rewardFunction(oldState, action)
                timeStep = [oldState, action, newState, reward] 
                replayBuffer = memory(replayBuffer, timeStep)
                if len(replayBuffer) >= self.numMiniBatch:
                    miniBatch = random.sample(replayBuffer, self.numMiniBatch)
                    states, actions, nextStates, rewards = list(zip(*miniBatch))
                    tarActor = lambda state: approximatePolicyEvaluation(state, actorModel)
                    tarCritic = lambda state, action: approximateQTarget(state, action, criticModel)
                    QLoss, criticModel = trainCritic(states, actions, nextStates, rewards, tarActor, tarCritic, criticModel)
                    gradientEvaCritic = lambda state, action: gradientPartialActionFromQEvaluation(state, action, criticModel)
                    gradientQPartialActorParameter, actorModel = trainActor(states, evaActor, gradientEvaCritic, actorModel)
                if self.isTerminal(oldState):
                    break
                oldState = newState

                if episodeIndex % 10 == 0:
                    self.render(oldState)
            print(timeStepIndex)
        return actorModel, criticModel

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)
    
    numActionSpace = 2
    numStateSpace = 24
    actionLow = np.array([-8, -8])
    actionHigh = np.array([8, 8])
    actionRatio = (actionHigh - actionLow) / 2.

    numActorFC1Unit = 128
    numActorFC2Unit = 128
    numActorFC3Unit = 128
    numActorFC4Unit = 128
    numCriticFC1Unit = 300
    numCriticFC2Unit = 300
    numCriticFC3Unit = 128
    numCriticFC4Unit = 128
    learningRateActor = 0.0001
    learningRateCritic = 0.0003
    l2DecayCritic = 0.00001

    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    softReplaceRatio = 0.001

    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            gradientQPartialAction_ = tf.placeholder(tf.float32, [None, numActionSpace], name="gradientQPartialAction_")

        with tf.variable_scope("evaluationHidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            evaFullyConnected1_ = tf.layers.dense(inputs = state_, units = numActorFC1Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            evaFullyConnected2_ = tf.layers.dense(inputs = evaFullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            evaFullyConnected3_ = tf.layers.dense(inputs = evaFullyConnected2_, units = numActorFC3Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            evaFullyConnected4_ = tf.layers.dense(inputs = evaFullyConnected3_, units = numActorFC4Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            evaActionActivation_ = tf.layers.dense(inputs = evaFullyConnected4_, units = numActionSpace, activation = tf.nn.tanh, kernel_initializer = initWeight, bias_initializer = initBias )
            
        with tf.variable_scope("targetHidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            tarFullyConnected1_ = tf.layers.dense(inputs = state_, units = numActorFC1Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            tarFullyConnected2_ = tf.layers.dense(inputs = tarFullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            tarFullyConnected3_ = tf.layers.dense(inputs = tarFullyConnected2_, units = numActorFC3Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            tarFullyConnected4_ = tf.layers.dense(inputs = tarFullyConnected3_, units = numActorFC4Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias )
            tarActionActivation_ = tf.layers.dense(inputs = tarFullyConnected4_, units = numActionSpace, activation = tf.nn.tanh, kernel_initializer = initWeight, bias_initializer = initBias )
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaAction_ = tf.multiply(evaActionActivation_, actionRatio, name = 'evaAction_')
            tarAction_ = tf.multiply(tarActionActivation_, actionRatio, name = 'tarAction_')
            gradientQPartialActorParameter_ = tf.gradients(ys = evaAction_, xs = evaParams_, grad_ys = gradientQPartialAction_, name = 'gradientQPartialActorParameter_')

        with tf.variable_scope("train"):
            #-learningRate for ascent
            trainOpt_ = tf.train.AdamOptimizer(-learningRateActor, name = 'adamOpt_').apply_gradients(zip(gradientQPartialActorParameter_, evaParams_))
        actorInit = tf.global_variables_initializer()
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineDDPG', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace]), name='action_')
            QTarget_ = tf.placeholder(tf.float32, [None, 1], name="QTarget_")

        with tf.variable_scope("evaluationHidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numCriticFC1Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias ))
            evaFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = evaFullyConnected1_, units = numCriticFC2Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias ))
            evaStateFC2ToFullyConnected3Weights_ = tf.get_variable(name='evaStateFC2ToFullyConnected3Weights', shape = [numCriticFC2Unit, numCriticFC3Unit], initializer = initWeight)
            evaActionToFullyConnected3Weights_ = tf.get_variable(name='evaActionToFullyConnected3Weights', shape = [numActionSpace, numCriticFC3Unit], initializer = initWeight)
            evaFullyConnected3Bias_ = tf.get_variable(name = 'evaFullyConnected3Bias', shape = [numCriticFC3Unit], initializer = initBias)
            evaFullyConnected3_ = tf.nn.relu(tf.matmul(evaFullyConnected2_, evaStateFC2ToFullyConnected3Weights_) + tf.matmul(action_, evaActionToFullyConnected3Weights_) +
                    evaFullyConnected3Bias_ )
            evaFullyConnected4_ = tf.layers.batch_normalization(tf.layers.dense(inputs = evaFullyConnected3_, units = numCriticFC4Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias ))
            evaQActivation_ = tf.layers.dense(inputs = evaFullyConnected3_, units = 1, activation = None, kernel_initializer = initWeight, bias_initializer = initBias )

        with tf.variable_scope("targetHidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numCriticFC1Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias ))
            tarFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = tarFullyConnected1_, units = numCriticFC2Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias ))
            tarStateFC2ToFullyConnected3Weights_ = tf.get_variable(name='tarStateFC2ToFullyConnected3Weights', shape = [numCriticFC2Unit, numCriticFC3Unit], initializer = initWeight)
            tarActionToFullyConnected3Weights_ = tf.get_variable(name='tarActionToFullyConnected3Weights', shape = [numActionSpace, numCriticFC3Unit], initializer = initWeight)
            tarFullyConnected3Bias_ = tf.get_variable(name = 'tarFullyConnected3Bias', shape = [numCriticFC3Unit],initializer = initBias)
            tarFullyConnected3_ = tf.nn.relu(tf.matmul(tarFullyConnected2_, tarStateFC2ToFullyConnected3Weights_) + tf.matmul(action_, tarActionToFullyConnected3Weights_) +
                    tarFullyConnected3Bias_ )
            tarFullyConnected4_ = tf.layers.batch_normalization(tf.layers.dense(inputs = tarFullyConnected3_, units = numCriticFC4Unit, activation = tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias ))
            tarQActivation_ = tf.layers.dense(inputs = tarFullyConnected3_, units = 1, activation = None, kernel_initializer = initWeight, bias_initializer = initBias )
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaQ_ = tf.multiply(evaQActivation_, 1, name = 'evaQ_')
            tarQ_ = tf.multiply(tarQActivation_, 1, name = 'tarQ_')
            diff_ = tf.subtract(QTarget_, evaQ_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
            gradientQPartialAction_ = tf.gradients(evaQ_, action_, name = 'gradientQPartialAction_')
            criticLossSummary = tf.summary.scalar("CriticLoss", loss_)
        with tf.variable_scope("train"):
            trainOpt_ = tf.contrib.opt.AdamWOptimizer(weight_decay = l2DecayCritic, learning_rate = learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineDDPG', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)    
   
    xBoundary = [0, 360]
    yBoundary = [0, 360]
    checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary)
    
    initSheepPosition = np.array([180, 180]) 
    initWolfPosition = np.array([180, 180])
    initSheepVelocity = np.array([0, 0])
    initWolfVelocity = np.array([0, 0])
    initSheepPositionNoise = np.array([30, 30])
    initWolfPositionNoise = np.array([0, 0])
    sheepPositionAndVelocityReset = ag.SheepPositionAndVelocityReset(initSheepPosition, initSheepVelocity, initSheepPositionNoise, checkBoundaryAndAdjust)
    wolfPositionAndVelocityReset = ag.WolfPositionAndVelocityReset(initWolfPosition, initWolfVelocity, initWolfPositionNoise, checkBoundaryAndAdjust)
    
    numOneAgentState = 12
    positionIndex = [0, 1]
    velocityIndex = [2, 3]
    sheepVelocitySpeed = 9
    sheepActionFrequency = 1
    wolfVelocitySpeed = 5
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
    possibleWolfSubtleties = [3.3]
    conditions = it.product(possibleAgentIds, possibleWolfSubtleties)
    transitionFunctions = [env.TransitionFunction(agentIds, sheepIndexOfId, wolfIndexOfId, wolfSubtlety, 
        sheepPositionAndVelocityReset, wolfPositionAndVelocityReset, sheepPositionAndVelocityTransation, wolfPositionAndVelocityTransation) 
        for agentIds, wolfSubtlety in conditions]
    
    minDistance = 12
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
    deathPenalty = -1
    rewardDecay = 0.99
    rewardFunctions = [reward.RewardFunctionTerminalPenalty(agentIds, sheepIndexOfId, wolfIndexOfId, numOneAgentState, positionIndex,
        aliveBouns, deathPenalty, isTerminal) for agentIds, isTerminal in zip(possibleAgentIds, isTerminals)] 
    
    actionNoise = np.array([3, 3])
    noiseDecay = 0.9999
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)

    memoryCapacity = 100000
    memory = Memory(memoryCapacity)

    numMiniBatch = 128
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, numMiniBatch)
    trainActor = TrainActorTensorflow(actorWriter, numMiniBatch) 

    maxTimeStep = 200
    maxEpisode = 100000
    deepDeterministicPolicyGradient = OnlineDeepDeterministicPolicyGradient(maxEpisode, maxTimeStep, numMiniBatch, transitionFunctions, isTerminals, addActionNoise,
            rewardFunctions, render)
   
    trainedActorModel, trainedCriticModel = deepDeterministicPolicyGradient(actorModel, criticModel, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget,
            gradientPartialActionFromQEvaluation, memory, trainCritic, trainActor)

if __name__ == "__main__":
    main()



