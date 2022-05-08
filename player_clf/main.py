import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount
from tpot.export_utils import set_param_recursive
from sklearn.metrics import accuracy_score, f1_score

class ImpPlayerClassifier:

    def train_clf(self, features, target):
        exported_pipeline = make_pipeline(
            ZeroCount(),
            MLPClassifier(alpha=0.0001, learning_rate_init=0.01)
        )
        set_param_recursive(exported_pipeline.steps, 'random_state', 42)
        exported_pipeline.fit(features, target)
        return exported_pipeline

    def predict(self, model, features):
        return model.predict(features)
    
    def score(self, predictions, target):
        return {
            'accuracy': accuracy_score(target, predictions),
            'f1': f1_score(target, predictions, average='macro')
        }

train_x = np.load('data/X_train.npz')['arr_0']
train_y = np.load('data/y_train.npz')['arr_0']
val_x = np.load('data/X_validation.npz')['arr_0']
val_y = np.load('data/y_validation.npz')['arr_0']
test_x = np.load('data/X_test.npz')['arr_0']
test_y = np.load('data/y_test.npz')['arr_0']
X_train = np.concatenate((train_x, val_x))
y_train = np.concatenate((train_y, val_y))
print(X_train.shape, y_train.shape)

ipc_obj = ImpPlayerClassifier()
model = ipc_obj.train_clf(X_train, y_train)
pickle.dump(model, open('model/model.pkl', 'wb'))
model = pickle.load(open('model/model.pkl', 'rb'))
pred_y = ipc_obj.predict(model, test_x)
print(ipc_obj.score(pred_y, test_y))
