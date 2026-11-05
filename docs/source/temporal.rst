.. _tempfs:

Temporal Fuzzy Sets
===================

Temporal fuzzy sets (TFS) generalize fuzzy sets with a temporal variable that
modulates the membership values; a comprehensive explanation is in [Kiah]_.

They live in :mod:`ebdai.temporal`, not in :mod:`ex_fuzzy`: Ex-Fuzzy removed
its temporal module in 3.3.1, and BDI keeps it as an expansion on top of the
public ``ex_fuzzy`` API. Everything else in this guide composes with them.

A temporal set wraps an ordinary set and multiplies its membership by one value
per time moment. ``temporalFS.type()`` reports the family of the wrapped set
(``t1``, ``t2`` or ``gt2``), so the rest of BDI treats it as a set of that
family, and ``temporalFS.temporal_type()`` returns the temporal flavour from
``TMP_FUZZY_SETS``.

Discretizing time
=================

Temporal sets need the temporal variable discretized into moments. With a time
of day in the data, define the cut points of each moment::

    cut_points_morning = ['00:00:00', '10:00:00']
    cut_points_daytime = ['11:00:00', '19:00:00']
    cutpoints_evening = ['20:00:00', '23:00:00']

``ebdai.temporal`` provides the helpers that were previously in
``ex_fuzzy.utils``::

    temporal_boolean_markers = temporal.temporal_cuts(
        X_total, cutpoints=[cut_points_morning, cut_points_daytime, cutpoints_evening],
        time_resolution='hour')
    time_moments = np.array([temporal.assign_time(a, temporal_boolean_markers)
                             for a in range(X_total.shape[0])])

To keep the moments balanced, assemble the partitions first::

    partitions, partition_markers = temporal.temporal_assemble(
        X_total, y_total, temporal_moments=temporal_boolean_markers)
    X_train, X_test, y_train, y_test = partitions
    train_markers, test_markers = partition_markers

Building temporal partitions
============================

Given the time moments and the ordinary fuzzy partitions of every variable,
turn them into temporal fuzzy partitions::

    temp_partitions = temporal.create_tempVariables(
        X_total_array, time_moments, precomputed_partitions)

``create_multi_tempVariables`` builds one partition per moment instead, when
each moment deserves its own fuzzy sets.

Training and evaluating
=======================

:class:`ebdai.temporal.TemporalFuzzyRulesClassifier` fits one rule base per
time moment with the genetic search of :class:`ex_fuzzy.BaseFuzzyRulesClassifier`::

    from ebdai import temporal

    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
    fl_classifier = temporal.TemporalFuzzyRulesClassifier(
        nRules=nRules, nAnts=nAnts, linguistic_variables=temp_partitions,
        fuzzy_type=fz_type_studied, verbose=True, tolerance=0.001, n_class=2)
    fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=pop_size,
                      time_moments=train_time_moments)

    temporal.eval_temporal_fuzzy_model(
        fl_classifier, X_train, y_train, X_test, y_test,
        time_moments=train_time_moments, test_time_moments=test_time_moments,
        plot_rules=False, print_rules=True, plot_partitions=False)

Predictions need the moment of each sample::

    predictions = fl_classifier.predict(X_test, test_time_moments)

The fitted ``fl_classifier.rule_base`` is a
:class:`ebdai.temporal.temporalMasterRuleBase`, which holds one rule base per
time moment. Its firing strengths only consider the rules of the sample's own
moment. ``fl_classifier.eval_performance`` is an ordinary
:class:`ex_fuzzy.eval_rules.evalRuleBase`; the rule base binds the training
moments, so the usual metrics, such as ``association_degree()``, work
unchanged.

References
==========

.. [Kiah] Kiani, M., Andreu-Perez, J., & Hagras, H. (2022). A Temporal Type-2
   Fuzzy System for Time-dependent Explainable Artificial Intelligence. *IEEE
   Transactions on Artificial Intelligence*.
