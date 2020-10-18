# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
from pacman import GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractorDigit(datum)


    "*** YOUR CODE HERE ***"

    # imageWidth = DIGIT_DATUM_WIDTH
    # imageHeight = DIGIT_DATUM_HEIGHT

    add_array = []
    max_count = 0
    count = 1
    region = 0
    region_count = 0
    final = 0
    region_counter = 0

    featCount = 1

    for i in range(DIGIT_DATUM_WIDTH):
        # add_array.append(features[(i, 0)])
        add_array.append(0)

    for y in range(DIGIT_DATUM_HEIGHT):

        max_count = max(add_array)

        if max_count >= region_count:
            region_count = max_count

        final = max(region_counter, final)

        count = 1
        region = 0
        region_counter = 0
        isOne = 0

        for x in range(DIGIT_DATUM_WIDTH):
            pixel = features[(x, y)]

            if pixel == 1:
                if region == 1:
                    region_counter += 1
                    add_array[x] = count
                    region = 0
                    isOne = 1
                else:
                    add_array[x] = count
                    region = 0
                    isOne = 1
                    # region_counter += 1
            else:
                if isOne == 0 and add_array != 1:
                    count += 1
                    add_array[x] = 0
                    region = 1
                else:
                    add_array[x] = 0
                    region = 1
                    isOne = 0

    feature_counter = 1

    if final <= 0:
        features['1'] = 1
        features['2'] = 1
        features['3'] = 0
        features['5'] = 1
        features['7'] = 1
        features['4'] = 0
        features['6'] = 0
        features['8'] = 0
        features['9'] = 0
    else:
        features['4'] = 1
        features['6'] = 1
        features['8'] = 1
        features['9'] = 1
        features['1'] = 0
        features['2'] = 0
        features['3'] = 1
        features['5'] = 0
        features['7'] = 1

    for x in range(DIGIT_DATUM_WIDTH):
        NumArea = check1 = area = check2 = 0
        for y in range(DIGIT_DATUM_HEIGHT):
            if features[(x, y)] > 0:
                if check1 == 0:
                    check1 = 1
            if features[(x, y)] == 0:
                if check1 == 1:
                    if area == 0:
                        area = 1
            if features[(x, y)] > 0:
                if check1 == 1:
                    if area == 1:
                        check2 = 1
            if features[(x, y)] == 0:
                if check1 == 1:
                    if area == 1:
                        if check2 == 1:
                            NumArea += 1
                            check1 = area = check2 = 0
            if NumArea > 0:
                features[feature_counter] = 1
            else:
                features[feature_counter] = 0
            feature_counter += 1

    # region_count -= 1

    # if region_count == 1:
    #     features['1'] = 1
    # elif region_count == 2:
    #     features['4'] = 1
    #     features['5'] = 1
    #     features['6'] = 0
    #     features['9'] = 1
    # else:
    #     features['4'] = 1

    # for y in range(DIGIT_DATUM_HEIGHT):
    #
    #     for x in range(DIGIT_DATUM_WIDTH):
    #         pixel = features[(x, y)]
    #         print pixel,
    #     print ""
    #
    # print final

    # newMat = util.Counter()
    #
    # for i in range(DIGIT_DATUM_WIDTH):
    #
    #     for j in range(DIGIT_DATUM_HEIGHT):
    #
    #         if features[(i, j)] == 0:
    #             newMat[(i, j)] = 0
    #         else:
    #             if newMat[(i-1, j)] != newMat[(i, j-1)]:
    #
    #                 newMat[(i, j)] = min(newMat[(i-1, j)], newMat[(i, j-1)])
    #
    #             else:
    #
    #                 if features[i - newMat[(i-1, j)], j - newMat[(i - 1, j)]] == 1:
    #
    #                     newMat[(i, j)] = newMat[(i - 1, j)] + 1
    #                 else:
    #                     newMat[(i, j)] = newMat[(i - 1, j)]

    return features


def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()


def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()


def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    "*** YOUR CODE HERE ***"

    successor = state.generateSuccessor(0, action)
    foodCount = successor.getNumFood()
    curCapsules = successor.getCapsules()
    capsuleCount = len(successor.getCapsules())
    score = successor.getScore()
    ghostCount = len(successor.getGhostPositions())
    win = successor.isWin()
    lose = successor.isLose()

    pacmanPosition = successor.getPacmanPosition()

    currentFood = state.getFood()

    width = len(currentFood[:])
    height = len(currentFood[0])
    minDist = float("inf")

    for x in range(width):
        for y in range(height):
            if currentFood[x][y]:
                dist = util.manhattanDistance((x, y), pacmanPosition)
                if dist < minDist:
                    minDist = dist

    minGhostDist = float("inf")

    ghostList = []
    for g in successor.getGhostStates():
        ghostList.append(util.manhattanDistance(g.getPosition(),pacmanPosition))

    capDist = float("inf")
    for capsuleState in curCapsules:
        capDist = min(capDist, util.manhattanDistance(pacmanPosition, capsuleState))

    # features['foodCount'] = foodCount
    # features['capsuleCount'] = capsuleCount
    # features['score'] = score
    # features['ghostCount'] = ghostCount

    # ghostStates = successor.getGhostStates()
    # scaredGhosts = [gs for gs in ghostStates if gs.scaredTimer > 0]
    #
    # if len(scaredGhosts) > 0:
    #     features['scaredGhosts'] = 1
    # else:
    #     features['scaredGhosts'] = 0


    # features['capsuleCount'] = (1 / (1 + capsuleCount))
    # features['score'] = score
    # features['ghostCount'] = 1 / (1 + ghostCount)

    # if action == 'STOP':
    #     features['Stop'] = 1
    # else:
    #     features['Stop'] = 0

    # features['win'] = win
    # features['lose'] = lose
    # features['features'] = score + (1 / (1 + capsuleCount)) + (1 / (1 + ghostCount)) \
    #                              + (1 / (1 + minDist)) + (1 / (1 + minGhostDist))\
    #                              + (1 / (1 + capDist)) + foodCount

    # features['minDistanceToFood'] = 1 / (1 + minDist)
    # features['minDistanceToGhost'] = 1 / (1 + minGhostDist)
    # features['minDistanceToCapsule'] = 1 / (1 + capDist)

    # features['minDistanceToFood'] = minDist
    # features['minDistanceToGhost'] = minGhostDist
    # features['minDistanceToCapsule'] = capDist


    features['food'] = minDist

    if action == 'Stop':
        features['Stop'] = 1
    else:
        features['Stop'] = 0

    features['foodCount'] = foodCount
    features['ghost'] = min(ghostList)

    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features = basicFeatureExtractorDigit(datum)
    return features


def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features = basicFeatureExtractorFace(datum)
    return features


def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    # prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print rawTestData[i]
    #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None, self.width, self.height)
        for pix in pixels:
            try:
                # This is so that new features that you could define which
                # which are not of the form of (x,y) will not break
                # this image printer...
                x, y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image


def default(str):
    return str + ' [Default: %default]'


USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser

    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'),
                      choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'],
                      default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'],
                      default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False,
                      action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False,
                      action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"),
                      type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False,
                      action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if (options.data == "digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif (options.data == "faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif (options.data == "pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if (options.data == "digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if (options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif (options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" % options.smoothing
    elif (options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels, options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels, options.iterations)
    elif (options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print "using automatic tuning for MIRA"
            classifier.automaticTuning = True
        else:
            print "using default C=0.001 for MIRA"
    elif (options.classifier == 'minicontest'):
        import minicontest

        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl', 'pacmandata/food_validation.pkl', 'pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl', 'pacmandata/stop_validation.pkl', 'pacmandata/stop_test.pkl' ),
    'SuicideAgent': (
        'pacmandata/suicide_training.pkl', 'pacmandata/suicide_validation.pkl', 'pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': (
        'pacmandata/good_reflex_training.pkl', 'pacmandata/good_reflex_validation.pkl',
        'pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': (
        'pacmandata/contest_training.pkl', 'pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data
    numTraining = options.training
    numTest = options.test

    if (options.data == "pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or \
                       MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or \
                         MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH,
                                               DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest, DIGIT_DATUM_WIDTH,
                                                 DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (
        100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if ((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1, label2)
        if (options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print string3
        printImage(features_odds)

    if ((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ===" % l)
            printImage(features_weights)


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runClassifier(args, options)
