import numpy as np


def OpenSetROC(score, galLabels, probLabels, farPoints=None, rankPoints=None):
    """ DIR, FAR, thresholds = OpenSetROC( score, galLabels, probLabels, farPoints, rankPoints )

        A function for open-set identification ROC evaluation.

        Inputs:
            score: score matrix with rows corresponding to gallery and columns probe.
            galLabels: a vector containing class labels of each gallery sample,
                  corresponding to rows of the score matrix.
            probLabels: a vector containing class labels of each probe sample,
                  corresponding to columns of the score matrix.
            farPoints: interested points of the false accept rates for
                  evaluation. Optional.
            rankPoints: interested points of the matching ranks. Optional.

        Outputs:
            DIR: detection and identification rates, with rows corresponding to
                  ranks and columins corresponding to FARs
            FAR: false accept rates. This is preferred than farPoints for
              performance plot, because with insufficient number of impostor probes,
              some small far points may cannot be reached, and the corresponding
              returned FARs will be set to zero.
            thresholds: decision thresholds used to generate DIR and FAR.

        Example:
            farPoints = [0, kron(10.^(-4:-1), 1:9), 1]
            rankPoints = [1:10, 20:10:100]
            [DIR, FAR, thresholds] = OpenSetROC( score, galLabels, probLabels, farPoints, rankPoints )
            figure surf(FAR * 100, rankPoints, DIR * 100)
            xlabel( 'False Accept Rate (#)' )
            ylabel( 'Rank' )
            zlabel( 'Detection and Identification Rate (#)' )
            title( 'Open-set Identification Performance' )
            figure semilogx( far*100, DIR(10,:)*100, 'r-o' ) grid on
            xlabel( 'False Accept Rate (#)' )
            ylabel( 'Detection and Identification Rate (#)' )
            title( 'Open-set Identification ROC Curve at Rank 10' )
            [~, farIndex] = ismember(0.01, farPoints)
            figure semilogx( rankPoints, DIR(:,farIndex)*100, 'r-o' ) grid on
            xlabel( 'Rank' )
            ylabel( 'Detection and Identification Rate (#)' )
            title( 'Open-set Identification CMC Curve at FAR = 0.01' )
    """

    ## preprocess
    if rankPoints is None:
        # rankPoints = [1:10, 20:10:100]
        rankPoints = np.array(list(range(10)) + list(range(20, 101, 10)))
        rankPoints = rankPoints[rankPoints <= len(galLabels)]
    rankPoints = np.array(rankPoints)

    galLabels = np.array(galLabels)
    probLabels = np.array(probLabels)

    galLabels = galLabels.reshape([-1, 1])
    probLabels = probLabels.reshape([1, -1])

    binaryLabels = galLabels == probLabels # match / non-match labels corresponding to the score matrix

    t = np.any(binaryLabels == True, axis=0) # determine whether a probe belongs to the gallery
    genProbIndex = t==True # seperate the probe set into genuine probe set and impostor probe set
    impProbIndex = t==False
    Ngen = np.sum(genProbIndex)
    Nimp = np.sum(impProbIndex)

    # set the number of false alarms
    farPoints = np.array(farPoints)
    if farPoints is None:
        falseAlarms = np.arange(Nimp+1)
    else:
        if np.any(farPoints < 0) or np.any(farPoints > 1):
            raise ('FAR should be in the range [0,1].')
        
        falseAlarms = np.round(farPoints * Nimp)

    falseAlarms -= 1
    ## get detection scores and matching ranks of each probe
    impScore = np.max(score[:, impProbIndex], axis=0) # maximum scores of each impostor probe
    impScore = -np.sort(-impScore)  # descending

    S = score[:, genProbIndex] # matching scores of each genuine probe
    sortedIndex = -np.argsort(-S, axis=0) # rank the score
    M = binaryLabels[:, genProbIndex] # match / non-match labels
    # clear binaryLabels
    S[M == False] = float('-Inf') # set scores of non-matches to -Inf
    # clear M
    # [genScore, genGalIndex] = max(S) # get maximum genuine score of the matches, as well as the location of the matches
    genScore = np.max(S, axis=0)
    genGalIndex = np.argmax(S, axis=0)
    # clear S
    # [probRanks, ~] = find( bsxfun(@eq, sortedIndex, genGalIndex) ) # get the matching ranks of each genuine probe, by finding the location of the matches in the sorted index
    probRanks = np.argmax(sortedIndex == genGalIndex, axis=0) # get the matching ranks of each genuine probe, by finding the location of the matches in the sorted index
    # clear sortedIndex

    ## compute thresholds
    isZeroFAR = (falseAlarms == 0)
    isOneFAR = (falseAlarms == Nimp)
    thresholds = np.zeros((len(falseAlarms.ravel())))
    thresholds[~isZeroFAR & ~isOneFAR] = impScore[ falseAlarms[~isZeroFAR & ~isOneFAR].astype(int) ] # use the sorted imporstor scores to generate the decision thresholds

    # when FAR=0, the decision threshold should be a bit larger than
    # impScore(1), because the decision is made by the ">=" operator
    highGenScore = genScore[genScore > impScore[0]]
    if len(highGenScore) == 0:
        thresholds[isZeroFAR] = impScore[0]
    else:
        thresholds[isZeroFAR] = (impScore[0] + np.min(highGenScore) ) / 2


    # when FAR = 1, the decision threshold should be the minimum score that can
    # also accept all genuine scores
    thresholds[isOneFAR] = np.min([impScore[-1], np.min(genScore)])

    ## evaluate
    genScore = genScore.reshape([-1,1])

    thresholds = thresholds.reshape([1,-1])
    probRanks = probRanks.reshape([-1,1])
    rankPoints = rankPoints.reshape([1,-1])

    T1 = genScore >= thresholds # compare genuine scores to the decision thresholds
    T2 = probRanks <= rankPoints # compare the genuine probe matching ranks to the interested rank points
    # detection and identification should be both satisfied
    # T = T1.reshape([Ngen, 1, -1]).repeat(T2.shape[1], axis=1) & T2.reshape([Ngen, -1, 1]).repeat(T1.shape[1], axis=2)
    T = []
    for i in range(Ngen):
        T += [T1[i].reshape([1,-1]) & T2[i].reshape([-1,1])]
    T = np.array(T)
    DIR = np.squeeze( np.mean(T, axis=0) ) # average over all genuine probes 
    FAR = falseAlarms / Nimp

    return DIR, FAR, thresholds

