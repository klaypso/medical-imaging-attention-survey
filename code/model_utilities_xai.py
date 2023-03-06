# Imports
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2

# PyTorch Imports
import torch
import torch.nn as nn

# Captum Imports
from captum.attr import DeepLift, LRP
from captum.attr._utils.custom_modules import Addition_Module
from captum.attr._utils.lrp_rules import EpsilonRule, PropagationRule

# Transformer xAI Imports
from transformer_explainability_utils.ViT_explanation_generator import LRP as DeiT_LRP
from transformer_explainability_utils.ViT_explanation_generator import Baselines as DeiT_Baselines

# Project Imports
from model_utilities_cbam import ChannelPool


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Class: CustomLRP
class CustomLRP(LRP):
    def _check_and_attach_rules(self) -> None:
        SUPPORTED_NON_LINEAR_LAYERS = [nn.ReLU, nn.Dropout, nn.Tanh, nn.Sigmoid]
        SUPPORTED_LAYERS_WITH_RULES = {
            nn.MaxPool1d: EpsilonRule,
            nn.MaxPool2d: EpsilonRule,
            nn.MaxPool3d: EpsilonRule,
            nn.Conv2d: EpsilonRule,
            nn.AvgPool2d: EpsilonRule,
            nn.AdaptiveAvgPool2d: EpsilonRule,
            nn.Linear: EpsilonRule,
            nn.BatchNorm2d: EpsilonRule,
            Addition_Module: EpsilonRule,
            ChannelPool: EpsilonRule,
            }

        for layer in self.layers:
            if hasattr(layer, "rule"):
                layer.activations = {}  # type: ignore
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
                pass
            elif type(layer) in SUPPORTED_LAYERS_WITH_RULES.keys():
                layer.activations = {}  # type: ignore
                layer.rule = SUPPORTED_LAYERS_WITH_RULES[type(layer)]()  # type: ignore
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
            elif type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                layer.rule = None  # type: ignore
            else:
                raise TypeError(
                    (
                        f"Module of type {type(layer)} has no rule defined and no"
                        "default rule exists for this module type. Please, set a rule"
                        "explicitly for this module and assure that it is appropriate"
                        "for this type of layer."
                    )
                )
    

    def _check_rules(self) -> None:
        for module in self.mod