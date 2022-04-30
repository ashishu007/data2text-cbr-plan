import argparse
import json
import numpy as np
import pyswarms as ps
from sklearn.metrics import log_loss

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-ent', '--ent', type=str, default='players', help='entity to rank', choices=['players', 'teams'])
args = arg_parser.parse_args()

# ent_abb_dict = {'p': 'player', 'tp': 'team_and_player', 'pp': 'player_and_player'}
ent_type = args.ent

print(args)
print('Loading data...')

data = json.load(open(f'ranking_outs/{ent_type}4pso_train.json'))
X = np.array([item['features'] for item in data])
y = np.array([item['label'] for item in data])
num_features = X.shape[1]

print(X.shape, y.shape, X.shape[0], num_features)

# Forward propagation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """

    # logits = logits_function(params)
    logits = X.dot(params)

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores)#, axis=1, keepdims=True)

    # Compute for the negative log likelihood

    # corect_logprobs = -np.log(probs[range(num_samples), y])
    # loss = np.sum(corect_logprobs) / num_samples
    
    loss = log_loss(y, probs)

    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
max_bound = np.ones(num_features)
min_bound = np.negative(max_bound)
bounds = (min_bound, max_bound)

# Call instance of PSO
dimensions = num_features
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)#, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=100)
np.save(f'ranking_outs/{ent_type}_weights.npy', pos)
