"""
This module contains the Check class check results from previous models against current model results.
"""

import os
import pickle
import time
from model.pyrice import PyRICE
from model.enumerations import *


class Check:
    """
    This class is used to check the current model results against the original model results.
    """

    def __init__(self, quick=False, save=False):
        """
        @param quick: Boolean
        @param save: Boolean
        """
        self.quick = quick
        self.start_time = time.time()

        if quick:
            self.srs = [0.270]
            self.mius = [2135]
            self.irstps = [0.015]
        else:
            self.srs = [0.270, 0.35]
            self.mius = [2135, 2070]
            self.irstps = [0.015, 0.07]

        self.spec = [ModelSpec.STANDARD,
                     ModelSpec.Validation_1,
                     ModelSpec.Validation_2]

        self.wf = [WelfareFunction.UTILITARIAN,
                   WelfareFunction.EGALITARIAN,
                   WelfareFunction.SUFFICIENTARIAN,
                   WelfareFunction.PRIORITARIAN]

        self.f_damage = [DamageFunction.NORDHAUS,
                         DamageFunction.NEWBOLD,
                         DamageFunction.WEITZMAN]

        self.dicts = []

        if save:
            self.save_my_pickle(file='new_data')

    def run_models(self):

        """
        Create and run several models.
        Return a list of results dictionaries.
        """

        self.dicts = []

        counter = 0
        max_runs = len(self.srs) * len(self.mius) * len(self.irstps) * len(self.spec) * len(self.wf) * \
                   len(self.f_damage)

        print_step = int(max_runs / 10)

        for spec in self.spec:
            for welfare in self.wf:
                for damage in self.f_damage:
                    for sr in self.srs:
                        for miu in self.mius:
                            for irstp in self.irstps:

                                counter += 1
                                if counter % print_step == 0 and counter != max_runs:
                                    print(f'Run #{counter}/{max_runs}')

                                m = PyRICE(model_specification=spec, damage_function=damage, welfare_function=welfare)
                                results = m(sr=sr, miu=miu, irstp=irstp)

                                self.dicts.append(results)

        print(f'Run #{counter}/{max_runs}\n')
        return self.dicts

    @staticmethod
    def load_my_pickle(folder='/testdata/', file='original_data'):
        """
        @param folder: string
        @param file: string
        @return:
            original_data: pickle
        """
        directory = os.getcwd()
        highest_directory = os.path.dirname(directory)
        original_pyrice_directory = highest_directory + "/verification/pyrice"

        with open(f'{original_pyrice_directory + folder}{file}.pickle', 'rb') as handle:
            original_data = pickle.load(handle)

        return original_data

    def save_my_pickle(self, folder='/testdata/', file='original_data'):
        """

        @param folder: string
        @param file: string
        """
        directory = os.getcwd()
        highest_directory = os.path.dirname(directory)
        original_pyrice_directory = highest_directory + "/verification/pyrice"

        results = self.run_models()
        modifier = '_quick' if self.quick else '_slow'

        with open(f'{original_pyrice_directory + folder}{file}{modifier}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __call__(self):
        """
        Checks whether the current model returns the same results as the original model by comparing current results
        with the saved results from the folder testdata.
        """

        modifier = '_quick' if self.quick else '_slow'
        original_data = self.load_my_pickle(file=f'original_data{modifier}')
        new_data = self.run_models()

        run_time = round(time.time() - self.start_time, 2)
        print(f'Run time: {run_time} seconds')

        is_identical = original_data == new_data

        print(f'\nOriginal and new results are identical: {is_identical}')

        modifier_ = "GOOD JOB! :D" if is_identical else "OH, NOOO! :("

        message = f"\n####################################################################################\n" \
                  "####################################################################################\n" \
                  "####################################################################################\n" \
                  f"################################# {modifier_} #####################################\n" \
                  "####################################################################################\n" \
                  "####################################################################################\n" \
                  "####################################################################################"

        print(message)
