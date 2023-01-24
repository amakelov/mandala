from mandala.all import *
from mandala.tests.utils import *


def test_deletion_bug():
    storage = Storage()

    @op
    def add(x: int, y: int) -> int:
        return x + y

    with storage.run():
        x = 23
        for i in range(10):
            y = add(x, i)
            z = add(y, i)

    with storage.run():
        x = 23
        for i in range(10):
            with storage.delete():
                y = add(x, i)
                z = add(y, i)


def test_rf():
    from typing import List, Tuple
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification, load_digits
    from pathlib import Path
    import numpy as np
    from numpy import ndarray

    Config.enable_ref_magics = True
    Config.warnings = False
    storage = Storage()

    @op
    def generate_data() -> Tuple[ndarray, ndarray]:
        return load_digits(n_class=2, return_X_y=True)

    @op
    def train_and_eval_tree(
        X, y, seed, max_depth=1
    ) -> Tuple[DecisionTreeClassifier, float]:
        print("WHAT")
        tree = DecisionTreeClassifier(
            random_state=seed, max_depth=max_depth, max_features=1
        ).fit(X, y)
        return tree, round(accuracy_score(y_true=y, y_pred=tree.predict(X)), 2)

    @op
    def eval_forest(trees: List[DecisionTreeClassifier], X, y) -> float:
        majority_vote = (
            np.array([tree.predict(X) for tree in trees]).mean(axis=0) >= 0.5
        )
        return round(accuracy_score(y_true=y, y_pred=majority_vote), 2)

    @superop
    def train_forest(X, y, n_trees) -> List[DecisionTreeClassifier]:
        trees = []
        for i in range(n_trees):
            tree, acc = train_and_eval_tree(X, y, seed=i)
            if acc > 0.8:
                trees.append(tree)
        return trees

    with storage.run():
        X, y = generate_data()
        for n_trees in (5, 10, 15):
            trees = train_forest(X, y, n_trees)
            forest_acc = eval_forest(trees, X, y)

    with storage.run():
        X, y = generate_data()
        for n_trees in (5, 10, 15):
            with storage.delete():
                trees = train_forest(X, y, n_trees)
                forest_acc = eval_forest(trees, X, y)


def test_add_input_plus_deps():
    storage = Storage(deps_root=Path().absolute())

    @op
    def inc(x: int) -> int:
        return x + 1

    with storage.run():
        x = inc(23)

    @op
    def inc(x: int, y=1) -> int:
        return x + y

    with storage.run():
        x = inc(23, y=2)

    @op
    def inc(x: int, y=1) -> int:
        return x + y + 1

    with storage.run():
        x = inc(23, y=2)
