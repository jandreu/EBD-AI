"""
Temporal fuzzy sets for BDI.

Temporal-aware fuzzy sets, temporal rule bases, ``TemporalFuzzyRulesClassifier``
and the helpers that turn time stamps into temporal partitions. Ex-Fuzzy
removed its temporal module in 3.3.1; BDI keeps it here, on top of the public
``ex_fuzzy`` API, so the rebase can follow upstream without losing it.

Main Components:
    - TMP_FUZZY_SETS: Enumeration of temporal fuzzy set types
    - temporalFS: Temporal fuzzy set implementation
    - temporalFuzzyVariable: Temporal fuzzy variable container
    - temporalMasterRuleBase: Temporal rule base for time-dependent reasoning
    - TemporalFuzzyRulesClassifier: genetic classifier over temporal sets
    - eval_temporal_fuzzy_model: training and test report for a temporal model
    - create_tempVariables, create_multi_tempVariables, temporal_cuts,
      temporal_assemble: helpers that build temporal partitions from time stamps

A temporal set reports the type of the set it multiplies in time (``t1``,
``t2`` or ``gt2``) from ``type()``, so the rest of BDI can treat it as an
ordinary set of that family; ``temporal_type()`` says which temporal flavour
it is.
"""
import copy
import enum

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from ex_fuzzy import eval_rules as evr
from ex_fuzzy import evolutionary_fit as evf
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import rules as rl
from ex_fuzzy import utils
from ex_fuzzy import vis_rules


class TMP_FUZZY_SETS(enum.Enum):
    """Legacy temporal-only enum retained with stable member values."""

    temporal = 1
    temporal_t2 = 2
    temporal_gt2 = 3


# Compatibility alias for callers that imported the formerly generated enum.
NEW_FUZZY_SETS = fs.FUZZY_SETS

#: Which temporal flavour wraps each family of set.
_TEMPORAL_TYPE_OF = {fs.FUZZY_SETS.t1: TMP_FUZZY_SETS.temporal,
                     fs.FUZZY_SETS.t2: TMP_FUZZY_SETS.temporal_t2,
                     fs.FUZZY_SETS.gt2: TMP_FUZZY_SETS.temporal_gt2}
### DEFINE THE FUZZY SET ####
class temporalFS(fs.FS):
    """
    Class to implement temporal fuzzy sets.
    """

    def __init__(self, std_fuzzy_set: fs.FS, conditional_variable: np.array) -> None:
        """
        Creates a temporal fuzzy set.

        Args:
            std_fuzzy_set: FS. Standard fuzzy set that contains the non-temporal aware memberhsip function.
            conditional_variable: np.array. The variable that expresses the different temporal moments. Shape (time discrete moments, ).
        """
        self.std_set = std_fuzzy_set
        self.tmp_function = conditional_variable
        self.time = None
        self.name = std_fuzzy_set.name

        if std_fuzzy_set.type() == fs.FUZZY_SETS.t1:
            self.membership_parameters = std_fuzzy_set.membership_parameters
        elif std_fuzzy_set.type() == fs.FUZZY_SETS.t2:
            self.secondMF_upper = std_fuzzy_set.secondMF_upper
            self.secondMF_lower = std_fuzzy_set.secondMF_lower
        else:
            self.secondary_memberships = std_fuzzy_set.secondary_memberships
            self.alpha_cuts = std_fuzzy_set.alpha_cuts


    def membership(self, input: np.array, time: int=None) -> np.array:
        """
        Computes the membership of each sample and in each time for the fs.

        Args:
            input: array temporal_time x samples.
                :time: int. Time moment to compute the membership. If none, looks for a fixed time

        Returns:
            array temporal_time x samples.
        """
        if time is None:
            if self.time is None:
                raise ValueError('Temporal fuzzy set has no fixed time. Please, fix a time or provide a time to compute the membership.')
            time = self.time

        return self.std_set.membership(input) * self.tmp_function[time]


    def type(self) -> fs.FUZZY_SETS:
        """
        Returns the type of the set this one multiplies in time: t1, t2 or gt2.

        Ex-Fuzzy removed the temporal members of ``FUZZY_SETS``, so a temporal set reports the
        family of the set it wraps and the rest of BDI handles it as one of those.
        """
        return self.std_set.type()


    def temporal_type(self) -> TMP_FUZZY_SETS:
        """
        Returns which temporal flavour this set is.
        """
        return _TEMPORAL_TYPE_OF[self.std_set.type()]
    

    def inside_type(self) -> fs.FUZZY_SETS:
        """
        Returns the type of the og fuzzy set computed before the time dependency.
        """
        return self.std_set.type()
    

    def fix_time(self, time: int) -> None:
        """
        Fixes the time of the temporal fuzzy set.

        Args:
            time: int. Time moment to fix.

        Returns:
            FS. Fuzzy set with the fixed time.
        """
        self.time = time



#### DEFINE THE FUZZY VARIABLE ####
class temporalFuzzyVariable(fs.fuzzyVariable):
    """
    Class to implement a temporal fuzzy variable.
    """

    def __init__(self, name: str, fuzzy_sets: list[temporalFS], units: str = None) -> None:
        """
        Creates a temporal fuzzy variable.

        Args:
            str: name of the variable.
            fuzzy_sets: list of the fuzzy sets pre-time dependencies.
            units: units of the variable, if any.
        """
        self.linguistic_variables = []
        self.name = name
        self.units = units
        self.time = None
        for ix, fs in enumerate(fuzzy_sets):
            self.linguistic_variables.append(fs)

        self.fs_type = self.linguistic_variables[0].type()


    def fix_time(self, time: int) -> None:
        """
        Fixes the time of the temporal fuzzy variable.

        Args:
            time: int. Time moment to fix.
        """
        self.time = time


    def compute_memberships(self, x: np.array, time: int=None) -> dict:
        """
        Computes the membership to each of the FS in the fuzzy variables.

        Args:
            x: numeric value or array. Computes the membership to each of the IVFS in the fuzzy variables.
            time: int. Time moment to compute the membership.

        Returns:
            list of floats. Membership to each of the FS in the fuzzy variables.
        """
        if time is None:
            if self.time is None:
                raise ValueError('Temporal fuzzy variable has no fixed time. Please, fix a time or provide a time to compute the membership.')
            time = self.time

        res = []

        for fuzzy_set in self.linguistic_variables:
            res.append(fuzzy_set.membership(x, time))

        return res
    

    def n_time_moments(self) -> int:
        """
        Returns the number of time moments of the temporal fuzzy variable.

        Returns:
            int. Number of time moments of the temporal fuzzy variable.
        """
        return self.linguistic_variables[0].tmp_function.shape[0]


#### DEFINE THE RULE BASE WITH TEMPORAL DEPENDENCIES ####
class temporalMasterRuleBase(rl.MasterRuleBase):
    """
    This class is a temporal extension of the MasterRuleBase class. It includes a list of
    rule bases for each time step.
    """
    def __init__(self, rule_base: list[rl.MasterRuleBase], time_step_names: list[str]=None):
        """
        Constructor of the temporalMasterRuleBase class.

        Args:
            rule_base: list of rule bases.
            time_steps: number of time steps.
        """
        super().__init__(rule_base)
        self.time_steps = np.arange(len(rule_base))
        self.time_mrule_bases = rule_base
        #: Time moment of each sample, bound by :meth:`bind_time_moments` for the generic evaluators.
        self._time_moments = None

        if time_step_names is None:
            self.time_step_names = [str(x) for x in self.time_steps]
        else:
            self.time_step_names = time_step_names
        
        for ix, mrb in enumerate(rule_base):
            for jx, rb in enumerate(mrb):
                for antecedent in rb.antecedents:
                    antecedent.fix_time(ix)
        
        self.antecedents = rule_base[0].antecedents
                     

    def bind_time_moments(self, time_moments: list[int]) -> 'temporalMasterRuleBase':
        """
        Fix the time moment of each sample, for the generic evaluators.

        ``ex_fuzzy.eval_rules.evalRuleBase`` calls ``compute_firing_strengths(X, precomputed_truth=...)``
        without time moments, because Ex-Fuzzy's own temporal support was removed upstream. Binding them
        once here lets the temporal methods be called with the ordinary signatures until rebound.

        Args:
            time_moments: time moment of each sample, or None to clear the binding.

        Returns:
            this rule base, so the call can be chained.
        """
        self._time_moments = None if time_moments is None else np.asarray(time_moments)
        return self


    def _resolve_time_moments(self, time_moments) -> np.array:
        """Uses the given time moments, or the bound ones when the caller passes none."""
        if time_moments is not None:
            return np.asarray(time_moments)
        if self._time_moments is None:
            raise ValueError('This temporal rule base has no time moments. Pass them, or call '
                             'bind_time_moments first.')
        return self._time_moments


    def add_rule(self, rule: rl.RuleSimple, consequent: int, time_step: int) -> None:
        """
        Adds a rule to the rule base of the given consequent.

        Args:
            rule: rule to add.
            consequent: index of the rule base to add the rule.
            time_step: time step of the rule base to add the rule.
        """
        self.time_mrule_bases[time_step][consequent].add_rule(rule)


    def get_rulebase_matrix(self) -> list[np.array]:
        """
        Returns a list with the rulebases for each antecedent in matrix format.

        Returns:
            list with the rulebases for each antecedent in matrix format.
        """
        return [a.get_rulebase_matrix() for x in self.time_mrule_bases for a in x]


    def get_scores(self) -> np.array:
        """
        Returns the dominance score for each rule in all the rulebases.

        Returns:
            array with the dominance score for each rule in all the rulebases.
        """
        res = []
        for rule_bases in self.time_mrule_bases:    
            for rb in rule_bases:
                res.append(rb.scores())

        res = [x for x in res if len(x) > 0]

        if len(res) == 0:
            return np.array([])
        return np.concatenate(res, axis=0)


    def compute_firing_strengths(self, X: np.array, time_moments: list[int]=None, precomputed_truth=None) -> np.array:
        """
        Computes the firing strength of each rule for each sample.

        Rules only fire for the samples of their own time moment.

        Args:
            X: array with the values of the inputs.
            time_moments: time moment of each sample. Defaults to the moments bound with
                :meth:`bind_time_moments`.

        Returns:
            array with the firing strength of each rule for each sample.
        """
        aux = []
        time_moments = self._resolve_time_moments(time_moments)

        for time_moment, rule_bases in enumerate(self.time_mrule_bases):
            actual_moment = np.equal(time_moments, time_moment)
            for rule_base in rule_bases:
                if len(rule_base) == 0:
                    continue
                memberships = rule_base.compute_rule_antecedent_memberships(X)
                mask = actual_moment.reshape((-1,) + (1,) * (memberships.ndim - 1))
                aux.append(memberships * mask)

        if len(aux) == 0:
            return np.zeros((X.shape[0], 0))

        # Firing strengths shape: samples x rules (x 2) (last is iv dimension)
        return np.concatenate(aux, axis=1)


    def compute_association_degrees(self, X: np.array, time_moments: list[int]=None, precomputed_truth=None) -> np.array:
        """
        Computes the association degree of each rule for each sample, weighting the firing strengths by the dominance scores.

        Args:
            X: array with the values of the inputs.
            time_moments: time moment of each sample. Defaults to the moments bound with
                :meth:`bind_time_moments`.

        Returns:
            array with the association degree of each rule for each sample.
        """
        firing_strengths = self.compute_firing_strengths(X, time_moments)
        if firing_strengths.shape[1] == 0:
            return firing_strengths

        scores = self.get_scores()
        while scores.ndim < firing_strengths.ndim - 1:
            scores = scores[..., None]

        association_degrees = scores * firing_strengths
        while association_degrees.ndim > 2:
            association_degrees = np.mean(association_degrees, axis=-1)

        return association_degrees


    def winning_rule_predict(self, X: np.array, time_moments: list[int]=None, precomputed_truth=None, out_class_names=False) -> np.array:
        """
        Returns the predicted consequent for each sample. Takes into account dominance scores if already computed.

        Args:
            X: array with the values of the inputs.
            time_moments: time moment of each sample. Defaults to the moments bound with
                :meth:`bind_time_moments`.
            out_class_names: if True, returns the consequent names instead of their indexes.

        Returns:
            array with the consequent of the winning rule for each sample (-1 or 'Unknown' if no rule fires).
        """
        consequents = []
        for mrb in self.time_mrule_bases:
            for jx, rb in enumerate(mrb):
                consequent = mrb.consequent_names[jx] if out_class_names else jx
                consequents.extend([consequent] * len(rb))

        winning_rules, _winning_association_degrees = self._winning_rules(X, time_moments, allow_unknown=self.allow_unknown)
        unknown = 'Unknown' if out_class_names else -1

        return np.array([consequents[ix] if ix != -1 else unknown for ix in winning_rules])


    def add_rule_base(self, rule_base: rl.RuleBase, time: int) -> None:
        """
        Adds a rule base to the list of rule bases.

        Args:
            rule_base: rule base to add.
        """
        self.time_mrule_bases[time].add_rule_base(rule_base)


    def print_rules(self, return_rules=False, bootstrap_results=True) -> None:
        """
        Print all the rules for all the consequents.

        Args:
            return_rules: if True, the rules are returned as a string instead of printed.
            bootstrap_results: if True, the bootstrap statistics of the rules are included.
        """
        res = ''
        for zx, time in enumerate(self.time_mrule_bases):
            res += 'Rules for time step: ' + self.time_step_names[zx] + '\n'
            res += '----------------\n'
            for ix, ruleBase in enumerate(time):
                try:
                    res += 'Consequent: ' + str(time.consequent_names[ix]) + '\n'
                    res += ruleBase.print_rules(True, bootstrap_results)
                    res += '\n'
                except IndexError:
                    pass # We did not have rules for this consequent
            res += '----------------\n'

        if return_rules:
            return res
        else:
            print(res)


    def get_rules(self) -> list[rl.RuleSimple]:
        """
        Returns a list with all the rules.

        Returns:
            list with all the rules.
        """
        return [rule for mruleBase in self.rule_bases for rule in mruleBase.get_rules()]


    def fuzzy_type(self) -> fs.FUZZY_SETS:
        """
        Returns the correspoing type of the RuleBase using the enum type in the fuzzy_sets module.

        Returns:
            the corresponding fuzzy set type of the RuleBase.
        """
        return self.time_mrule_bases[0].rule_bases[0].fuzzy_type()


    def purge_rules(self, tolerance=0.001) -> None:
        """
        Delete the roles with a dominance score lower than the tolerance.

        Args:
            tolerance: tolerance to delete the rules.
        """
        for mruleBase in self.time_mrule_bases:
            mruleBase.purge_rules(tolerance)


    def __getitem__(self, item: int) -> rl.RuleBase:
        """
        Returns the corresponding time rulebase.

        Args:
            item: index of the rulebase.

        Returns:
            the corresponding rulebase.
        """
        return self.time_mrule_bases[item]


    def __len__(self) -> int:
        """
        Returns the number of rule bases.
        """
        return len(self.time_mrule_bases)


    def get_consequents(self) -> list[int]:
        """
        Returns a list with the consequents of each rule base.

        Returns:
            list with the consequents of each rule base.
        """
        return sum([x.get_consequents() for ix, x in enumerate(self.time_mrule_bases)], [])
    

    def get_rulebases(self) -> list[rl.RuleBase]:
        """
        Returns a list with all the rules.

        Returns:
            list
        """
        rule_bases = []

        for ruleBase in self.time_mrule_bases:
            for rule in ruleBase:
                rule_bases.append(rule)
        
        return rule_bases
    

    def _winning_rules(self, X: np.array, temporal_moments: list[int]=None, precomputed_truth=None, allow_unknown=True) -> tuple[np.array, np.array]:
        """
        Returns the winning rule for each sample. Takes into account dominance scores if already computed.

        Args:
            X: array with the values of the inputs.
            temporal_moments: time moment of each sample. Defaults to the moments bound with
                :meth:`bind_time_moments`.
            allow_unknown: if True, samples that fire no rule get -1 as winning rule.

        Returns:
            the index of the winning rule and its association degree for each sample.
        """
        association_degrees = self.compute_association_degrees(X, temporal_moments)
        if association_degrees.shape[1] == 0:
            return np.full(X.shape[0], -1), np.zeros(X.shape[0])

        winning_rules = np.argmax(association_degrees, axis=1)
        # The winner's association degree, as the generic rule base does.
        winning_association_degrees = np.max(association_degrees, axis=1)

        if allow_unknown:
            winning_rules[np.max(association_degrees, axis=1) == 0.0] = -1

        return winning_rules, winning_association_degrees
        

#### DEFINE THE FUZZY CLASSIFIER USING TEMPORAL FUZZY SETS ####
class TemporalFuzzyRulesClassifier(evf.BaseFuzzyRulesClassifier):
    """
    Class that is used as a classifier for a fuzzy rule based system with time dependencies. Supports precomputed and optimization of the linguistic variables.
    """

    def __init__(self,  nRules: int = 30, nAnts: int = 4, fuzzy_type: fs.FUZZY_SETS = None, tolerance: float = 0.0,
                 n_linguistic_variables: int = 0, verbose=False, linguistic_variables: list[fs.fuzzyVariable] = None,
                 domain: list[float] = None, n_class: int=None, precomputed_rules: rl.MasterRuleBase =None, runner: int=1) -> None:
        """
        Inits the optimizer with the corresponding parameters.


        kwargs:
            - n_classes: number of classes to predict. Default deduces from data.

        Args:
            nRules: number of rules to optimize.
            nAnts: max number of antecedents to use.
            fuzzy_type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
            tolerance: tolerance for the dominance score of the rules.
            n_linguistic_variables: number of linguistic variables per antecedent.
            verbose: if True, prints the progress of the optimization.
            linguistic_variables: list of fuzzyVariables type. If None (default) the optimization process will init+optimize them.
            domain: list of the limits for each variable. If None (default) the classifier will compute them empirically.
        """
        super().__init__(nRules=nRules, nAnts=nAnts, fuzzy_type=fuzzy_type, tolerance=tolerance, 
                         n_linguistic_variables=n_linguistic_variables, verbose=verbose, linguistic_variables=linguistic_variables, 
                         domain=domain, n_class=n_class, precomputed_rules=precomputed_rules, runner=runner)
    

    def _contruct_tempRuleBase(self, problems, best_individuals):
        ruleBase_temp = []

        for problem, subject in zip(problems, best_individuals):
            ruleBase_time_moment = problem._construct_ruleBase(subject, 
                                                                self.fuzzy_type)
            ruleBase_temp.append(ruleBase_time_moment)

        return ruleBase_temp


    def _fix_time(self, lvs: list[temporalFuzzyVariable], time: int) -> None:
        """
        Fixes the time of the temporal fuzzy variables.

        Args:
            lvs: list of temporal fuzzy variables.
            time: integer. Time moment to fix.

        Returns:
            None. The time is fixed.
        """
        for lv in lvs:
            lv.fix_time(time)


    def fit(self, X: np.array, y: np.array, n_gen:int=50, pop_size:int=10, time_moments: np.array=None, checkpoints:int=0):
        """
        Fits a fuzzy rule based classifier using a genetic algorithm to the given data.

        Args:
            X: numpy array samples x features
            y: labels. integer array samples (x 1)
            n_gen: integer. Number of generations to run the genetic algorithm.
            pop_size: integer. Population size for each gneration.
            time_moments: array of integers. Time moments associated to each sample (when temporal dependencies are present)

        Returns:
            None. The classifier is fitted to the data.
        """
        if self.lvs is None:
            raise ValueError('TemporalFuzzyRulesClassifier needs temporal linguistic variables. '
                             'Build them with utils.create_tempVariables and pass them as linguistic_variables.')

        problems = []
        for ix in range(len(np.unique(time_moments))):
            X_problem = X[time_moments == ix]
            y_problem = y[time_moments == ix]

            time_lvs = [copy.deepcopy(aux) for aux in self.lvs]
            self._fix_time(time_lvs, ix)
            problem = evf.FitRuleBase(X_problem, y_problem, nRules=self.nRules, nAnts=self.nAnts,
                                linguistic_variables=time_lvs, domain=self.domain, tolerance=self.tolerance,
                                n_classes=self.nclasses_, thread_runner=self.thread_runner)

            problems.append(problem)

        

        # pymoo is imported only here. Wrapping first raises an actionable
        # error when it is missing.
        pymoo_problems = [evf.ev_backends.as_pymoo_problem(problem) for problem in problems]
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.optimize import minimize
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PolynomialMutation

        best_individuals = []
        self.performance = {}
        for time, problem in enumerate(problems):
            algorithm = GA(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=.3, eta=3.0),
            mutation=PolynomialMutation(eta=7.0),
            eliminate_duplicates=True)

            if checkpoints > 0:
                if self.verbose:
                    print('=================================================')
                    print('n_gen  |  n_eval  |     f_avg     |     f_min    ')
                    print('=================================================')
                algorithm.setup(pymoo_problems[time], seed=33, termination=('n_gen', n_gen)) # 33? Soon...
                for k in range(n_gen):
                    algorithm.next()
                    res = algorithm
                    if self.verbose:
                        print('%-6s | %-8s | %-8s | %-8s' % (res.n_gen, res.evaluator.n_eval, round(res.pop.get('F').mean(), 8), round(res.pop.get('F').min(), 8)))
                    if k % checkpoints == 0:
                        with open("checkpoint_" + str(time) + '_' + str(algorithm.n_gen), "w") as f:
                            pop = algorithm.pop
                            fitness_last_gen = pop.get('F')
                            best_solution_arg = np.argmin(fitness_last_gen)
                            best_individual = pop.get('X')[best_solution_arg, :]

                            rule_base = problem._construct_ruleBase(
                                best_individual, self.fuzzy_type)
                            eval_performance = evr.evalRuleBase(
                                rule_base, np.array(X), y)
                            
                            eval_performance.add_full_evaluation()  
                            # self.rename_fuzzy_variables() This wont work on checkpoints!
                            rule_base.purge_rules(self.tolerance)
                            checkpoint_rules = rule_base.print_rules(True)
                            f.write(checkpoint_rules)     

            else:
                res = minimize(pymoo_problems[time],
                            algorithm,
                            # termination,
                            ("n_gen", n_gen),
                            copy_algorithm=False,
                            save_history=False,
                            verbose=self.verbose)

            pop = res.pop
            fitness_last_gen = pop.get('F')
            best_solution = np.argmin(fitness_last_gen)
            best_individual = pop.get('X')[best_solution, :]
            best_individuals.append(best_individual)

            if self.verbose:
                print('Rule based fit for time ' + str(time) + ' completed.')

        
            self.performance[time] = 1 - fitness_last_gen[best_solution]

        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        
        ruleBase_temp = self._contruct_tempRuleBase(problems, best_individuals)

        self.rule_base = temporalMasterRuleBase(ruleBase_temp)

        self.eval_performance = evr.evalRuleBase(self.rule_base.bind_time_moments(time_moments), np.array(X), y)

        self.eval_performance.add_full_evaluation()  
        self.rename_fuzzy_variables()
        self.rule_base.purge_rules(self.tolerance)
        self.n_features_in_ = self.X.shape[1]
        return self


    def predict(self, X: np.array, time_moments: list[int] = None) -> np.array:
        """
        Returns the predicted class index for each sample.

        Args:
            X: np array samples x features.
            time_moments: list of integers. Time moments associated to each sample (when temporal dependencies are present)

        Returns:
            np array samples (x 1) with the predicted class.
        """
        return self.forward(X, time_moments)


    def forward(self, X: np.array, time_moments: list[int] = None) -> np.array:
        """
        Returns the predicted class for each sample.

        Args:
            X: np array samples x features.
            time_moments: list of integers. Time moments associated to each sample (when temporal dependencies are present)

        Returns:
            np array samples (x 1) with the predicted class.
        """
        try:
            X = X.values  # If X was a numpy array
        except AttributeError:
            pass
        
        return self.rule_base.winning_rule_predict(X, time_moments)



    def plot_fuzzy_variables(self) -> None:
        """
        Plot the fuzzy partitions in each fuzzy variable, before their time dependencies.
        """
        fuzzy_variables = self.rule_base.rule_bases[0].antecedents

        for fv in fuzzy_variables:
            std_sets = [fuzzy_set.std_set for fuzzy_set in fv.linguistic_variables]
            vis_rules.plot_fuzzy_variable(fs.fuzzyVariable(fv.name, std_sets, units=fv.units))


    def get_rulebase(self) -> list[np.array]:
        """
        Get the rulebase obtained after fitting the classifier to the data.

        Returns:
            a matrix format for the rulebase.
        """
        return self.rule_base.get_rulebase_matrix()


def eval_temporal_fuzzy_model(fl_classifier: evf.BaseFuzzyRulesClassifier, X_train: np.array, y_train: np.array,
                     X_test: np.array, y_test: np.array, time_moments: list[int] = None, test_time_moments: list[int] = None,
                     plot_rules=True, print_rules=True, plot_partitions=True, return_rules=False, print_accuracy=True, print_matthew=True) -> None:
    """
    Function that evaluates a fuzzy rule based model and optionally plots its fuzzy partitions.

    Args:
        fl_classifier: Fuzzy rule based model.
        X_train: Training data.
        y_train: Training labels.
        X_test: Test data.
        y_test: Test labels.
        plot_rules: Deprecated compatibility argument; ignored.
        print_rules: If True, it prints the rules.
        plot_partitions: If True, it plots the fuzzy partitions.

    Returns:
        None
    """

    if print_accuracy:
      print('ACCURACY')
      print('Train performance: ' +
            str(np.mean(np.equal(y_train, fl_classifier.forward(X_train, time_moments)))))
      print('Test performance: ' +
            str(np.mean(np.equal(y_test, fl_classifier.forward(X_test, test_time_moments)))))
      print('------------')
    if print_matthew:
      print('MATTHEW CORRCOEF')
      print('Train performance: ' +
            str(matthews_corrcoef(y_train, fl_classifier.forward(X_train, time_moments))))
      print('Test performance: ' +
            str(matthews_corrcoef(y_test, fl_classifier.forward(X_test, test_time_moments))))
      print('------------')

    time_moments = np.asarray(time_moments)
    test_time_moments = np.asarray(test_time_moments)
    for ix in np.unique(time_moments):
        train_mask = time_moments == ix
        test_mask = test_time_moments == ix
        X_aux = X_train.iloc[train_mask, :] if hasattr(X_train, 'iloc') else X_train[train_mask, :]
        X_aux_test = X_test.iloc[test_mask, :] if hasattr(X_test, 'iloc') else X_test[test_mask, :]

        y_aux = np.asarray(y_train)[train_mask]
        y_aux_test = np.asarray(y_test)[test_mask]

        if print_matthew:
            print('MOMENT ' + str(ix))
            print('------------')
            print('MATTHEW CORRCOEF')
            print('Train performance: ' +
                  str(matthews_corrcoef(y_aux, fl_classifier.forward(X_aux, np.array([ix] * X_aux.shape[0])))))
            print('Test performance: ' +
                  str(matthews_corrcoef(y_aux_test, fl_classifier.forward(X_aux_test, np.array([ix] * X_aux_test.shape[0])))))
            print('------------')

    if plot_partitions:
        fl_classifier.plot_fuzzy_variables()

    if print_rules or return_rules:
        res = fl_classifier.print_rules(True)
        if print_rules:
            print(res)
        if return_rules:
            return res



#### HELPERS THAT BUILD TEMPORAL PARTITIONS FROM TIME STAMPS ####
def construct_conditional_frequencies(X: np.array, discrete_time_labels: list[int], initial_ffss: list[fs.FS]):
    """
    Computes the conditional temporal function for a set of fuzzy sets according to their variation in time.

    Args:
        X: numpy array, shape samples x features.
        discrete_time_labels: discrete time labels.
        initial_ffss: initial fuzzy set list.

    Returns:
        conditional frequencies. Array shape (time steps, initial fuzzy sets)
    """
    obs = X.shape[0]
    discrete_time_labels = np.array(discrete_time_labels)
    memberships = np.zeros((obs, len(initial_ffss)))
    for ix, fset in enumerate(initial_ffss):
        if fset.type() == fs.FUZZY_SETS.t2:
            memberships[:, ix] = np.mean(fset.membership(X), axis=1)
        elif fset.type() == fs.FUZZY_SETS.gt2:
            memberships[:, ix] = np.mean(np.squeeze(fset._alpha_reduction(fset.membership(X))), axis=1)
        else:
            memberships[:, ix] = fset.membership(X)

    max_memberships = np.argmax(memberships, axis=1)
    res = np.zeros((len(np.unique(discrete_time_labels)), len(initial_ffss)))

    for time in range(len(np.unique(discrete_time_labels))):

        relevant_memberships = max_memberships[time == discrete_time_labels]
        fs_winner_counter = np.zeros(len(initial_ffss))
        for ix, fset in enumerate(initial_ffss):
            fs_winner_counter[ix] = np.sum(relevant_memberships == ix)

        res[time, :] = fs_winner_counter

    return res / (np.max(res, axis=0) + 1e-6)


def classify_temp(dates: pd.DataFrame, cutpoints: tuple[str, str], time: str) -> np.array:
    """
    Classifies a set of dates according to the temporal cutpoints. Uses {time} as a the time resolution.
    Returns an array where true values are those values contained between those two date points.

    Args:
        dates: data observations to cut.
        cutpoints: points to check.
        time: time field to use as the criteria.

    Returns:
        boolean array. True values are those contained between the cutpoints.
    """

    def extract_hour(row):
        return row.__getattribute__(time)

    hours = pd.to_datetime(dates['date']).apply(extract_hour)

    cutpoint_series_0 = pd.to_datetime(pd.Series([cutpoints[0]] * len(dates)))
    cutpoint_series_0.index = dates.index
    hours0 = cutpoint_series_0.apply(extract_hour)

    cutpoint_series_1 = pd.to_datetime(pd.Series([cutpoints[1]] * len(dates)))
    cutpoint_series_1.index = dates.index
    hours1 = cutpoint_series_1.apply(extract_hour)

    condicion1 = hours >= hours0
    condicion2 = hours <= hours1

    return np.array(np.logical_and(condicion1, condicion2))


def assign_time(a: np.array, observations: list[np.array]) -> int:
    """
    Assigns a temporal moment to a set of observations.

    Args:
        a: array of boolean values.
        observations: list of boolean arrays with the corresponding timestamps.

    Returns:
        the index of the correspondent time moment for the a-th observation.

    Raises:
        ValueError if a is not timestamped in any of the observation arrays.
    """
    for ix, obs in enumerate(observations):
        if obs[a]:
            return ix

    raise ValueError('No temporal moment assigned')


def create_tempVariables(X_train: np.array, time_moments: np.array,
                         precomputed_partitions: list[fs.fuzzyVariable]) -> list['temporalFuzzyVariable']:
    """
    Creates a list of temporal fuzzy variables.

    Args:
        X_train: numpy array, shape samples x features.
        time_moments: time moments. Array shape (samples,). Each value is an integer denoting the n-th time moment of that observation.
        precomputed_partitions: precomputed partitions for each feature.

    Returns:
        list of temporal fuzzy variables.
    """
    temp_partitions = []
    for ix in range(X_train.shape[1]):
        feat_conditional = construct_conditional_frequencies(X_train[:, ix], time_moments, initial_ffss=precomputed_partitions[ix])
        temp_fs_list = []
        for vl in range(feat_conditional.shape[1]):
            vl_temp_fs = temporalFS(precomputed_partitions[ix][vl], feat_conditional[:, vl])
            temp_fs_list.append(vl_temp_fs)

        temp_fs_variable = temporalFuzzyVariable(precomputed_partitions[ix].name, temp_fs_list)
        temp_partitions.append(temp_fs_variable)

    return temp_partitions


def create_multi_tempVariables(X_train: np.array, time_moments: np.array,
                               fuzzy_type: fs.FUZZY_SETS) -> list[list['temporalFuzzyVariable']]:
    """
    Creates a list of lists of temporal fuzzy variables, one per time moment, built from the data of that moment.

    (So, instead of having one variable for all time moments, you have a different one for each moment that
    represents the same idea.)

    Args:
        X_train: numpy array, shape samples x features.
        time_moments: time moments. Array shape (samples,). Each value is an integer denoting the n-th time moment of that observation.
        fuzzy_type: kind of fuzzy sets used to build the partitions of each moment.

    Returns:
        list of lists of temporal fuzzy variables.
    """
    temp_partitions = []

    unique_time_moments = np.unique(time_moments)
    for time in unique_time_moments:
        X_obs = X_train[time_moments == time, :]
        precomputed_partitions = utils.construct_partitions(X_obs, fuzzy_type)

        temp_partitions.append(create_tempVariables(X_obs, time_moments[time_moments == time], precomputed_partitions))

    return temp_partitions


def temporal_cuts(X: pd.DataFrame, cutpoints: list[tuple[str, str]], time_resolution: str='hour') -> list[np.array]:
    """
    Returns a list of boolean indexes for each temporal moment. Performs the cuts between time steps using the cutpoints list.

    Args:
        X: data observations to cut in temporal moments.
        cutpoints: list of tuples with the cutpoints for each temporal moment.
        time_resolution: time field to use as the criteria.

    Returns:
        list of boolean arrays. True values are those contained between the cutpoints in each moment.
    """
    res = []
    for ix, cutpoint in enumerate(cutpoints):
        observations = classify_temp(X, cutpoint, time=time_resolution)
        res.append(observations)

    return res


def temporal_assemble(X: np.array, y: np.array, temporal_moments: list[np.array]):
    """
    Assembles the data in the temporal moments in order to have partitions with balanced time moments in each one.

    Args:
        X: data observations.
        y: labels.
        temporal_moments: list of boolean arrays. True values are those contained between the cutpoints in each moment.

    Returns:
        tuple of lists of data and labels for each temporal moment.
            First tuple is: X_train, X_test, y_train, y_test
            Second tuple is: train temporal moments, test temporal moments.
    """
    moments_partitions = []
    train_temporal_boolean_markers = []
    test_temporal_boolean_markers = []
    train_counter = 0
    test_counter = 0

    for ix, temporal_moment in enumerate(temporal_moments):
        X_train, X_test, y_train, y_test = train_test_split(X[temporal_moment], y[temporal_moment], test_size=0.33, random_state=0)
        moments_partitions.append((X_train, X_test, y_train, y_test))

    if isinstance(X_train, (pd.core.series.Series, pd.DataFrame)):
        X_train = pd.concat([moments_partitions[ix][0] for ix in range(len(moments_partitions))])
        X_test = pd.concat([moments_partitions[ix][1] for ix in range(len(moments_partitions))])
        y_train = np.concatenate([moments_partitions[ix][2] for ix in range(len(moments_partitions))])
        y_test = np.concatenate([moments_partitions[ix][3] for ix in range(len(moments_partitions))])
    else:
        X_train = np.concatenate([moments_partitions[ix][0] for ix in range(len(moments_partitions))])
        X_test = np.concatenate([moments_partitions[ix][1] for ix in range(len(moments_partitions))])
        y_train = np.concatenate([moments_partitions[ix][2] for ix in range(len(moments_partitions))])
        y_test = np.concatenate([moments_partitions[ix][3] for ix in range(len(moments_partitions))])

    for ix, temporal_moment in enumerate(temporal_moments):
        # Believe, this makes sense to avoid rounding errors in the size of the final vector
        _, _, y_train0, y_test0 = train_test_split(X[temporal_moment], y[temporal_moment], test_size=0.33, random_state=0)

        train_moment_observations = np.zeros((X_train.shape[0]))
        train_moment_observations[train_counter:train_counter + len(y_train0)] = 1
        train_counter += len(y_train0)
        train_temporal_boolean_markers.append(train_moment_observations)

        test_moment_observations = np.zeros((X_test.shape[0]))
        test_moment_observations[test_counter:test_counter + len(y_test0)] = 1
        test_counter += len(y_test0)
        test_temporal_boolean_markers.append(test_moment_observations)

    return [X_train, X_test, y_train, y_test], [train_temporal_boolean_markers, test_temporal_boolean_markers]
