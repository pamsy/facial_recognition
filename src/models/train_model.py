from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def load_data():
    
def train_model():
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train_pca, y_train)
    model.score(X_valid_pca, y_valid)

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train_valid, y_train_valid, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))