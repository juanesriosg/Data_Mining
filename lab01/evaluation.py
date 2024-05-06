from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """

    """
        TP  = t t
        FP =  f t
        FN =  t f
    """
    precision = 0
    recall = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for index, feature in enumerate(expected_results):
        if expected_results[index] and actual_results[index]:
            TP += 1
        elif expected_results[index] and not actual_results[index]:
            FN += 1
        elif not expected_results[index] and actual_results[index]:
            FP += 1
        elif not expected_results[index] and not actual_results[index]:
            TN += 1

    precision = TP / (TP + FP) if TP != 0 else 0
    recall = TP / (TP + FN)  if TP != 0 else 0

    return (precision , recall)
    
def accuracy(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the accuracy of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The accuracy of the predicted results.
    """

    """
        TP  = t t
        FP =  f t
        FN =  t f
    """
    precision = 0
    recall = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for index, feature in enumerate(expected_results):
        if expected_results[index] and actual_results[index]:
            TP += 1
        elif expected_results[index] and not actual_results[index]:
            FN += 1
        elif not expected_results[index] and actual_results[index]:
            FP += 1
        elif not expected_results[index] and not actual_results[index]:
            TN += 1

    accuracy = (TP + TN)/(TP+TN+FP+FN)

    return accuracy
def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """

    p, r = precision_recall(expected_results, actual_results)
    acc = accuracy(expected_results, actual_results)

    return 2*r*p/(r+p) if r != 0 and p != 0 else acc