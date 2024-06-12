#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sklearn.datasets import make_classification
import pandas as pd
from random import random

NUM_CLASSES = 25

class_weights = [random() for _ in range(NUM_CLASSES)]

X, y = make_classification(n_samples=100000,
                           n_features=200,
                           n_informative=35,
                           n_redundant=65,
                           n_classes=NUM_CLASSES,
                           weights=class_weights,
                           random_state=42)

df = pd.DataFrame(X.round(3), columns=[f"V_{i}" for i in range(200)])
df['Class'] = y

os.makedirs('./datasets', exist_ok=True)
df.to_csv('./datasets/synthetic.csv', index=False)
