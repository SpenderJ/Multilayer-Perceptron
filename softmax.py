# coding: utf-8

import numpy as np


def softmax_crossentropy(logits, reference_answers):
    """
    Loss Function (Log softmax) which is better than softmax on his own.
    - Better numerical stability...
        Crossentropy from logits and ids of correct answers

        :param logits: data of shape [batch, n_classes]
        :param reference_answers: correct answers
        :return:
    """
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy


def grad_softmax_crossentropy(logits, reference_answers):
    """
    Crossentropy gradients from logits and ids of correct answers

    :param logits: data of shape [batch, n_classes]
    :param reference_answers: correct answers
    :return:
    """
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return (- ones_for_answers + softmax) / logits.shape[0]