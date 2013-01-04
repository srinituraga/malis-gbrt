"""
Testing for the tree module (sklearn.tree).
"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from nose.tools import assert_raises
from nose.tools import assert_true

from sklearn import tree
from sklearn import datasets

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_classification_toy():
    """Check classification on a toy dataset."""
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    assert_array_equal(clf.predict(T), true_result)

    # With subsampling
    clf = tree.DecisionTreeClassifier(max_features=1, random_state=1)
    clf.fit(X, y)

    assert_array_equal(clf.predict(T), true_result)


def test_regression_toy():
    """Check regression on a toy dataset."""
    clf = tree.DecisionTreeRegressor()
    clf.fit(X, y)

    assert_almost_equal(clf.predict(T), true_result)

    # With subsampling
    clf = tree.DecisionTreeRegressor(max_features=1, random_state=1)
    clf.fit(X, y)

    assert_almost_equal(clf.predict(T), true_result)


def test_xor():
    """Check on a XOR problem"""
    y = np.zeros((10, 10))
    y[:5, :5] = 1
    y[5:, 5:] = 1

    gridx, gridy = np.indices(y.shape)

    X = np.vstack([gridx.ravel(), gridy.ravel()]).T
    y = y.ravel()

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)

    clf = tree.DecisionTreeClassifier(max_features=1)
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)

    clf = tree.ExtraTreeClassifier()
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)

    clf = tree.ExtraTreeClassifier(max_features=1)
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)


def test_graphviz_toy():
    """Check correctness of graphviz output on a toy dataset."""
    clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=1)
    clf.fit(X, y)
    from StringIO import StringIO

    # test export code
    out = StringIO()
    tree.export_graphviz(clf, out_file=out)
    contents1 = out.getvalue()

    tree_toy = StringIO("digraph Tree {\n"
    "0 [label=\"X[0] <= 0.0000\\nerror = 0.5"
    "\\nsamples = 6\\nvalue = [ 3.  3.]\", shape=\"box\"] ;\n"
    "1 [label=\"error = 0.0000\\nsamples = 3\\nvalue = [ 3.  0.]\", shape=\"box\"] ;\n"
    "0 -> 1 ;\n"
    "2 [label=\"error = 0.0000\\nsamples = 3\\nvalue = [ 0.  3.]\", shape=\"box\"] ;\n"
    "0 -> 2 ;\n"
    "}")
    contents2 = tree_toy.getvalue()

    assert contents1 == contents2, \
        "graphviz output test failed\n: %s != %s" % (contents1, contents2)

    # test with feature_names
    out = StringIO()
    out = tree.export_graphviz(clf, out_file=out,
                               feature_names=["feature1", ""])
    contents1 = out.getvalue()

    tree_toy = StringIO("digraph Tree {\n"
    "0 [label=\"feature1 <= 0.0000\\nerror = 0.5"
    "\\nsamples = 6\\nvalue = [ 3.  3.]\", shape=\"box\"] ;\n"
    "1 [label=\"error = 0.0000\\nsamples = 3\\nvalue = [ 3.  0.]\", shape=\"box\"] ;\n"
    "0 -> 1 ;\n"
    "2 [label=\"error = 0.0000\\nsamples = 3\\nvalue = [ 0.  3.]\", shape=\"box\"] ;\n"
    "0 -> 2 ;\n"
    "}")
    contents2 = tree_toy.getvalue()

    assert contents1 == contents2, \
        "graphviz output test failed\n: %s != %s" % (contents1, contents2)

    # test improperly formed feature_names
    out = StringIO()
    assert_raises(IndexError, tree.export_graphviz,
                  clf, out, feature_names=[])


def test_iris():
    """Check consistency on dataset iris."""
    for c in ('gini', \
              'entropy'):
        clf = tree.DecisionTreeClassifier(criterion=c)\
              .fit(iris.data, iris.target)

        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.9, "Failed with criterion " + c + \
            " and score = " + str(score)

        clf = tree.DecisionTreeClassifier(criterion=c,
                                          max_features=2,
                                          random_state=1)\
              .fit(iris.data, iris.target)

        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.5, "Failed with criterion " + c + \
            " and score = " + str(score)


def test_boston():
    """Check consistency on dataset boston house prices."""
    for c in ('mse',):
        clf = tree.DecisionTreeRegressor(criterion=c)\
              .fit(boston.data, boston.target)

        score = np.mean(np.power(clf.predict(boston.data) - boston.target, 2))
        assert score < 1, "Failed with criterion " + c + \
            " and score = " + str(score)

        clf = tree.DecisionTreeRegressor(criterion=c,
                                         max_features=6,
                                         random_state=1)\
              .fit(boston.data, boston.target)

        #using fewer features reduces the learning ability of this tree,
        # but reduces training time.
        score = np.mean(np.power(clf.predict(boston.data) - boston.target, 2))
        assert score < 2, "Failed with criterion " + c + \
            " and score = " + str(score)


def test_probability():
    """Predict probabilities using DecisionTreeClassifier."""
    clf = tree.DecisionTreeClassifier(max_depth=1, max_features=1,
            random_state=42)
    clf.fit(iris.data, iris.target)

    prob_predict = clf.predict_proba(iris.data)
    assert_array_almost_equal(
        np.sum(prob_predict, 1), np.ones(iris.data.shape[0]))
    assert np.mean(np.argmax(prob_predict, 1)
                   == clf.predict(iris.data)) > 0.9

    assert_almost_equal(clf.predict_proba(iris.data),
                        np.exp(clf.predict_log_proba(iris.data)), 8)


def test_arrayrepr():
    """Check the array representation."""
    # Check resize
    clf = tree.DecisionTreeRegressor(max_depth=None)
    X = np.arange(10000)[:, np.newaxis]
    y = np.arange(10000)
    clf.fit(X, y)


def test_pure_set():
    """Check when y is pure."""
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [1, 1, 1, 1, 1, 1]

    clf = tree.DecisionTreeClassifier().fit(X, y)
    assert_array_equal(clf.predict(X), y)

    clf = tree.DecisionTreeRegressor().fit(X, y)
    assert_array_equal(clf.predict(X), y)


def test_numerical_stability():
    """Check numerical stability."""
    old_settings = np.geterr()
    np.seterr(all="raise")

    X = np.array(
       [[152.08097839, 140.40744019, 129.75102234, 159.90493774],
        [142.50700378, 135.81935120, 117.82884979, 162.75781250],
        [127.28772736, 140.40744019, 129.75102234, 159.90493774],
        [132.37025452, 143.71923828, 138.35694885, 157.84558105],
        [103.10237122, 143.71928406, 138.35696411, 157.84559631],
        [127.71276855, 143.71923828, 138.35694885, 157.84558105],
        [120.91514587, 140.40744019, 129.75102234, 159.90493774]])

    y = np.array(
        [1., 0.70209277, 0.53896582, 0., 0.90914464, 0.48026916,  0.49622521])

    dt = tree.DecisionTreeRegressor()
    dt.fit(X, y)
    dt.fit(X, -y)
    dt.fit(-X, y)
    dt.fit(-X, -y)

    np.seterr(**old_settings)


def test_importances():
    """Check variable importances."""
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=10,
                                        n_informative=3,
                                        n_redundant=0,
                                        n_repeated=0,
                                        shuffle=False,
                                        random_state=0)

    clf = tree.DecisionTreeClassifier(compute_importances=True)
    clf.fit(X, y)
    importances = clf.feature_importances_
    n_important = sum(importances > 0.1)

    assert_equal(importances.shape[0], 10)
    assert_equal(n_important, 3)

    X_new = clf.transform(X, threshold="mean")
    assert 0 < X_new.shape[1] < X.shape[1]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    assert_true(clf.feature_importances_ is None)


def test_error():
    """Test that it gives proper exception on deficient input."""
    # Invalid values for parameters
    assert_raises(ValueError,
                  tree.DecisionTreeClassifier(min_samples_leaf=-1).fit,
                  X, y)

    assert_raises(ValueError,
                  tree.DecisionTreeClassifier(max_depth=-1).fit,
                  X, y)

    assert_raises(ValueError,
                  tree.DecisionTreeClassifier(min_density=2.0).fit,
                  X, y)

    assert_raises(ValueError,
                  tree.DecisionTreeClassifier(max_features=42).fit,
                  X, y)

    # Wrong dimensions
    clf = tree.DecisionTreeClassifier()
    y2 = y[:-1]
    assert_raises(ValueError, clf.fit, X, y2)

    # Test with arrays that are non-contiguous.
    Xf = np.asfortranarray(X)
    clf = tree.DecisionTreeClassifier()
    clf.fit(Xf, y)
    assert_array_equal(clf.predict(T), true_result)

    # predict before fitting
    clf = tree.DecisionTreeClassifier()
    assert_raises(Exception, clf.predict, T)
    # predict on vector with different dims
    clf.fit(X, y)
    t = np.asarray(T)
    assert_raises(ValueError, clf.predict, t[:, 1:])

   # use values of max_features that are invalid
    clf = tree.DecisionTreeClassifier(max_features=10)
    assert_raises(ValueError, clf.fit, X, y)

    clf = tree.DecisionTreeClassifier(max_features=-1)
    assert_raises(ValueError, clf.fit, X, y)

    clf = tree.DecisionTreeClassifier(max_features="foobar")
    assert_raises(ValueError, clf.fit, X, y)

    tree.DecisionTreeClassifier(max_features="auto").fit(X, y)
    tree.DecisionTreeClassifier(max_features="sqrt").fit(X, y)
    tree.DecisionTreeClassifier(max_features="log2").fit(X, y)
    tree.DecisionTreeClassifier(max_features=None).fit(X, y)

    # predict before fit
    clf = tree.DecisionTreeClassifier()
    assert_raises(Exception, clf.predict_proba, X)

    clf.fit(X, y)
    X2 = [-2, -1, 1]  # wrong feature shape for sample
    assert_raises(ValueError, clf.predict_proba, X2)

    # wrong sample shape
    Xt = np.array(X).T

    clf = tree.DecisionTreeClassifier()
    clf.fit(np.dot(X, Xt), y)
    assert_raises(ValueError, clf.predict, X)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    assert_raises(ValueError, clf.predict, Xt)

    # wrong length of sample mask
    clf = tree.DecisionTreeClassifier()
    sample_mask = np.array([1])
    assert_raises(ValueError, clf.fit, X, y, sample_mask=sample_mask)

    # wrong length of X_argsorted
    clf = tree.DecisionTreeClassifier()
    X_argsorted = np.array([1])
    assert_raises(ValueError, clf.fit, X, y, X_argsorted=X_argsorted)


def test_min_samples_leaf():
    """Test if leaves contain more than leaf_count training examples"""
    X = np.asfortranarray(iris.data.astype(tree._tree.DTYPE))
    y = iris.target

    for tree_class in [tree.DecisionTreeClassifier, tree.ExtraTreeClassifier]:
        clf = tree_class(min_samples_leaf=5).fit(X, y)

        out = clf.tree_.apply(X)
        node_counts = np.bincount(out)
        leaf_count = node_counts[node_counts != 0]  # drop inner nodes

        assert np.min(leaf_count) >= 5


def test_pickle():
    import pickle

    # classification
    obj = tree.DecisionTreeClassifier()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(iris.data, iris.target)
    assert score == score2, "Failed to generate same score " + \
            " after pickling (classification) "

    # regression
    obj = tree.DecisionTreeRegressor()
    obj.fit(boston.data, boston.target)
    score = obj.score(boston.data, boston.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(boston.data, boston.target)
    assert score == score2, "Failed to generate same score " + \
            " after pickling (regression) "


def test_multioutput():
    """Check estimators on multi-output problems."""

    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [-2, 1],
         [-1, 1],
         [-1, 2],
         [2, -1],
         [1, -1],
         [1, -2]]

    y = [[-1, 0],
         [-1, 0],
         [-1, 0],
         [1, 1],
         [1, 1],
         [1, 1],
         [-1, 2],
         [-1, 2],
         [-1, 2],
         [1, 3],
         [1, 3],
         [1, 3]]

    T = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_true = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    # toy classification problem
    clf = tree.DecisionTreeClassifier()
    y_hat = clf.fit(X, y).predict(T)
    assert_array_equal(y_hat, y_true)
    assert_equal(y_hat.shape, (4, 2))

    proba = clf.predict_proba(T)
    assert_equal(len(proba), 2)
    assert_equal(proba[0].shape, (4, 2))
    assert_equal(proba[1].shape, (4, 4))

    log_proba = clf.predict_log_proba(T)
    assert_equal(len(log_proba), 2)
    assert_equal(log_proba[0].shape, (4, 2))
    assert_equal(log_proba[1].shape, (4, 4))

    # toy regression problem
    clf = tree.DecisionTreeRegressor()
    y_hat = clf.fit(X, y).predict(T)
    assert_almost_equal(y_hat, y_true)
    assert_equal(y_hat.shape, (4, 2))


def test_sample_mask():
    """Test sample_mask argument. """
    # test list sample_mask
    clf = tree.DecisionTreeClassifier()
    sample_mask = [1] * len(X)
    clf.fit(X, y, sample_mask=sample_mask)
    assert_array_equal(clf.predict(T), true_result)

    # test different dtype
    clf = tree.DecisionTreeClassifier()
    sample_mask = np.ones((len(X),), dtype=np.int32)
    clf.fit(X, y, sample_mask=sample_mask)
    assert_array_equal(clf.predict(T), true_result)


def test_X_argsorted():
    """Test X_argsorted argument. """
    # test X_argsorted with different layout and dtype
    clf = tree.DecisionTreeClassifier()
    X_argsorted = np.argsort(np.array(X).T, axis=1).T
    clf.fit(X, y, X_argsorted=X_argsorted)
    assert_array_equal(clf.predict(T), true_result)


if __name__ == "__main__":
    import nose
    nose.runmodule()
