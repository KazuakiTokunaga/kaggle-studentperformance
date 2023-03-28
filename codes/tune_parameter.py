import optuna
from codes import runner


def objective(trial):

    max_depth = trial.suggest_int("max_depth", 3, 9)
    min_child_weight = trial.suggest_float("min_child_weight", 0.1, 10)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.95)
    subsample = trial.suggest_float("subsample", 0.6, 0.95)

    adhoc_params = {
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'colsample_bytree': colsample_bytree,
        'subsample': subsample
    }

    run.validation(adhoc_params = adhoc_params)
    run.evaluate_validation()

    return scores[0]



if __name__=="__main__":

    load_options = {
        'sampling': 1000,
        'split_labels': True,
        'parquet': True
    }
    validation_options = {
        'n_fold': 2
    }
    model_options={
        'ensemble': False,
        'model': 'xgb',
        'param_file': 'params_xgb001_test.json'
    }

    run = runner.Runner(load_options = load_options, 
                        validation_options = validation_options,
                        model_options = model_options)
    run.load_dataset()
    run.engineer_features()
    run.delete_df_train()


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print(f"Best objective value: {study.best_value}")
    print(f"Best parameter: {study.best_params}")