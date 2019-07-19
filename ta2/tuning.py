from collections import defaultdict

from btb import HyperParameter
from btb.selection import UCB1
from btb.tuning import GP

from ta2.template import load_template


class SelectorTuner:

    def __init__(self, templates, data_augmentation):
        self.template_names = templates
        self.templates = dict()
        self.selector = UCB1(templates)
        self.scores = defaultdict(list)
        self.data_augmentation = data_augmentation

    @staticmethod
    def _get_tunables(tunable_hyperparameters):
        tunables = list()
        defaults = dict()
        for block_name, params in tunable_hyperparameters.items():
            for param_name, param_details in params.items():
                key = (block_name, param_name)
                param_type = param_details['type']
                param_type = 'string' if param_type == 'str' else param_type

                if param_type == 'bool':
                    param_range = [True, False]
                else:
                    param_range = param_details.get('range') or param_details.get('values')

                value = HyperParameter(param_type, param_range)
                tunables.append((key, value))
                defaults[key] = param_details['default']

        return tunables, defaults

    def propose(self):
        if len(self.templates) < len(self.template_names):
            template_name = self.template_names[len(self.templates)]
            template, tunable_hyperparameters = load_template(
                template_name, self.data_augmentation)
            tunables, proposal = self._get_tunables(tunable_hyperparameters)
            self.templates[template_name] = template, GP(tunables)
            default = True
        else:
            template_name = self.selector.select(self.scores)
            template, tuner = self.templates[template_name]
            proposal = tuner.propose(1)
            default = False

        return template_name, template, proposal, default

    def add(self, template_name, proposal, score):
        tuner = self.templates[template_name][1]
        tuner.add(proposal, score)
        self.scores[template_name].append(score)
