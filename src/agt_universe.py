# Data splitter.
# Takes in input:
# - n = total number of agents that exists,
# - k = total number of agents sharing data
import numpy as np
from utils import to_probability
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import copy
_debug_ = False

class AgentUniverse:
    # 1. Create agents partitions (of size n)
    # 2. Create aggregator involving only k agents:
    #    Every agents will compute their individual costs by using:
    #    a Its public data + The DP data of the selected k agents
    # 3. Aggregate

    def __init__(self, params: dict):
        '''
        :param params (key,val) described below
            - data:  The dataset
            - x-feat: An array of feature descriptions
            - y-feat: The feature used for classification/regression
            - n-agents:  The total number of agents in the universe
            - n-sharing-agents:  The number of cooperative agents (sharing their data privately)
            - model:  The learning model -- provides the class to the learning algorithm
            - private_model:  The private learning model
            - aggregator:    The aggregator used to aggregate private data
            - seed:  A seed used to initialize the random functions
            - eval-metric: Evaluation metric
        '''
        self.data = params['data']
        self.seed = params['seed']
        self.x_feat, self.y_feat, self.p_feat = params['x-feat'], params['y-feat'], params['p-feat']
        self.rnd = np.random.RandomState(params['seed'])
        self.force_run = False

        self.scale = params['normalize'] == 'scale'
        self.data_train, self.data_test = train_test_split(self.data, test_size=0.2, random_state=self.seed)
        self.data_train = self.data_train.reset_index()
        self.data_test = self.data_test.reset_index()

        self.n_agents = n = params['n-agents']
        self.n_sharers = k = params['n-sharing-agents']
        self.sharers = np.asarray([])
        # The aggregated parameters of the agent's learning model after fitting
        self.agts_fit_aggr_model = dict()

        if k == 0:
            # No private learning (no one shares)
            self.learn_privately = False
            self.aggregator = None
            self.agts_fit_aggr_model = None
        else:
            self.sharers = self.rnd.choice(n, size=k, replace=False)
            self.learn_privately = True
            private_model = params['private-model']
            self.agents_private_model = {i: copy.deepcopy(private_model) for i in range(n)}
            self.aggregator = params['aggregator']

        self.agents_data = dict()
        self.agents_public_model = {i: copy.deepcopy(params['model']) for i in range(n)}
        self.eval_metric = params['eval-metric']

        ## Tree partitioner
        self.is_generator = params['private-partitioner'] is not None
        self.partitioner = params['private-partitioner']
        self.generator = params['private-generator']


    def update(self, params: dict):
        if self.seed != params['seed']:
            self.__init__(params)
            return

        self.x_feat, self.y_feat, self.p_feat = params['x-feat'], params['y-feat'], params['p-feat']

        self.scale = params['normalize'] == 'scale'
        # if params['normalize'] == 'scale':
        #     self.data_train[self.x_feat] = scale(self.data_train[self.x_feat])
        #     self.data_test[self.x_feat] = scale(self.data_test[self.x_feat])
        #     self.force_run = True

        update = False
        n, k = params['n-agents'], params['n-sharing-agents']
        if self.n_agents != n:
            self.n_agents = n
            self.agents_data = dict()
            self.agents_public_model = {i: copy.deepcopy(params['model']) for i in range(n)}
            update = True
        if self.n_sharers != k:
            self.n_sharers = k
            self.sharers = np.asarray([])
            update = True

        # The aggregated parameters of the agent's learning model after fitting
        self.agts_fit_aggr_model = dict()

        if k == 0:
            # No private learning (no one shares)
            self.learn_privately = False
            self.aggregator = None
            self.agts_fit_aggr_model = None
        else:
            if update:
                self.sharers = self.rnd.choice(n, size=k, replace=False)
            self.learn_privately = True
            self.agents_private_model = {i: copy.deepcopy(params['private-model']) for i in range(n)}
            self.aggregator = params['aggregator']

        self.eval_metric = params['eval-metric']
        ## Tree partitioner
        self.is_generator = params['private-generator'] is not None
        self.partitioner = params['private-generator']

        print(self.sharers)

    ''' Partitions a dataset into N datasets to feed to different agents'''
    def allocate_data_to_agents(self, type='lin'):
        '''
            :param type: lin - linear space between attributes or
                         rnd - randomly distributed
        '''
        attr = self.p_feat
        if type == 'lin':
            means = np.linspace(np.min(self.data_train[[attr]].values),
                                np.max(self.data_train[[attr]].values),
                                self.n_agents + 2)[1:-1]
        elif type == 'rnd':
            means = np.random(np.min(self.data_train[[attr]].values),
                              np.max(self.data_train[[attr]].values),
                              self.n_agents)

        agt_data_rows = {i: [] for i in range(self.n_agents)}

        for index, row in self.data_train.iterrows():
            Pr = to_probability(1 / (abs(row[attr] - means)+0.000001))
            i = np.random.choice(range(self.n_agents), p=Pr)
            agt_data_rows[i].append(index)

        self.agents_data = \
            {i: self.data_train.iloc[agt_data_rows[i], :].reset_index(drop=True)
             for i in range(self.n_agents)}

        if _debug_:
            print('allocated data to agents')

        return self.agents_data

    def fit(self, hyperparams=None):
        X_features, y_features = self.x_feat, self.y_feat

        # Every agent learns on its public model:
        for i in self.agents_public_model:
            Di = self.agents_data[i]
            X, y = Di[X_features], Di[y_features]
            self.agents_public_model[i].fit(X, y, hyperparams)

        # Agents aggregated their model with private models
        if self.learn_privately:
            shared_models = []
            for i in range(self.n_agents):
                Di = self.agents_data[i]
                X, y = Di[X_features], Di[y_features]

                ##############
                # if _debug_:
                #     unique, counts = np.unique(y, return_counts=True)
                #     print('agent ', i, 'unique-count', list(zip(unique, counts)))
                ##############

                self.agents_private_model[i].fit(X, y, hyperparams)
                if i in self.sharers:
                    shared_models.append(self.agents_private_model[i])

            # aggregate all shared private models
            for i in range(self.n_agents):
                #self.agts_fit_aggr_model[i] = self.aggregator([self.agents_public_model[i]] + shared_models)
                self.agts_fit_aggr_model[i] = self.aggregator([self.agents_private_model[i]] + shared_models)
                self.agts_fit_aggr_model[i].aggregate()


    def _fit(self, hyperparams=None):
        X_features, y_features = self.x_feat, self.y_feat

        shared_data = {}
        for i in self.sharers:
            Tree = self.partitioner(self.agents_data[i], X_features, y_features, hp=hyperparams, seed=self.seed)
            shared_data[i] = self.generator(data=self.agents_data[i], partitioner=Tree,
                                                   eps=hyperparams['epsilon'], seed=self.seed)

        for i in range(self.n_agents):
            data =  self.agents_data[i].copy(deep=True)
            for j in [k for k in shared_data if k != i]:
                data = data.append(shared_data[j], sort=False)

            if self.scale:
                data[self.x_feat] = scale(data[self.x_feat])
            X, y = data[self.x_feat], data[self.y_feat]
            self.agents_public_model[i].fit(X, y, hyperparams)

    def eval(self, metric='mse'):
        X_features, y_features = self.x_feat, self.y_feat
        Xtest, ytest = self.data_test[X_features], self.data_test[y_features]
        if self.scale:
            Xtest = scale(Xtest)
        results = {metric: {}}

        for i in range(self.n_agents):
            if self.learn_privately and not self.is_generator:
                results[metric][i] = np.round(self.agts_fit_aggr_model[i].eval(Xtest, ytest, metric=metric), 8)
            else:
                results[metric][i] = np.round(self.agents_public_model[i].eval(Xtest, ytest, metric=metric), 8)
        return results

    def run(self, hyperparams_fit, metric='mse'):
        if len(self.agents_data) == 0 or self.force_run:
            self.allocate_data_to_agents()
            self.force_run = False

        if self.is_generator:
            self._fit(hyperparams_fit)
        else:
            self.fit(hyperparams_fit)

        return self.eval(metric=metric)

    def run_best(self, model, metric='mse'):
        Xtrain, ytrain = self.data_train[self.x_feat], self.data_train[self.y_feat]
        Xtest, ytest = self.data_test[self.x_feat], self.data_test[self.y_feat]
        model.fit(Xtrain, ytrain)
        print('model: w=', np.round(model.w))
        print('model: b=', np.round(model.b))
        return model.eval(Xtest, ytest, metric=metric)

