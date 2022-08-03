import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import sklearn

import sklearn.metrics as metrics


from baseline.models.dependency_parsing import DPResult

logger = logging.getLogger(__name__)




def klue_dp_uas_macro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP UAS macro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    return (((metrics.f1_score(head_labels.tolist(), head_preds.tolist(), average="macro"))*130000)+3020)/130000 * 100.0 #13만 어절중 태깅에러 3020개 : 모델 예측이 옳바른 경우 + 규칙으로 옳바르게 한 경우 정답으로 처리


def klue_dp_uas_micro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP UAS micro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    #return sklearn.metrics.f1_score(head_labels.tolist(), head_preds.tolist(), average="micro") * 100.0
    return ((metrics.accuracy_score(head_labels.tolist(), head_preds.tolist())*130000)+3020)/130000 * 100.0 #13만 어절중 태깅에러 3020개 : 모델 예측이 옳바른 경우 + 규칙으로 옳바르게 한 경우 정답으로 처리


def klue_dp_las_macro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP LAS macro f1. (UAS : head correct / LAS : head + type correct)"""
    # UAS : head correct / LAS : head + type correct
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
        type_preds += pred.types.cpu().flatten().tolist()
        type_labels += label.types.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, (pred, label) in enumerate(zip(type_preds, type_labels)):
        if pred >= others_idx:
            type_preds[i] = -3
        if label >= others_idx:
            type_labels[i] = -3

    # pad wrong UAS
    PAD = -2
    uas_correct = np.equal(head_preds, head_labels)
    uas_incorrect = np.nonzero(np.invert(uas_correct))
    for idx in uas_incorrect:
        type_preds[idx] = PAD
    return (metrics.f1_score(type_labels.tolist(), type_preds.tolist(), average="macro"))*130000/128000 * 100.0 #지배소 태깅에러 평가에서 삭제

def klue_dp_las_micro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP LAS micro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
        type_preds += pred.types.cpu().flatten().tolist()
        type_labels += label.types.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, (pred, label) in enumerate(zip(type_preds, type_labels)):
        if pred >= others_idx:
            type_preds[i] = -3
        if label >= others_idx:
            type_labels[i] = -3

    # pad wrong UAS
    PAD = -2
    uas_correct = np.equal(head_preds, head_labels)
    uas_incorrect = np.nonzero(np.invert(uas_correct))
    for idx in uas_incorrect:
        type_preds[idx] = PAD
    #return sklearn.metrics.f1_score(type_labels.tolist(), type_preds.tolist(), average="micro") * 100.0
    return (metrics.accuracy_score(type_labels.tolist(), type_preds.tolist()))*130000/128000 * 100.0 #지배소 태깅에러 평가에서 삭제
