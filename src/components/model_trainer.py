import os
import sys 
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from  sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
    try:
        logging.info("Split training and test data")
        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        models = {
            "RANDOM FOREST": RandomForestRegressor(),
            "DECISION TREE": DecisionTreeRegressor(),
            "GRADIENT BOOSTING": GradientBoostingRegressor(),
            "LINEAR REGRESSION": LinearRegression(),
            "K NEIGHBORS": KNeighborsRegressor(),   # ✅ changed
            "XGB REGRESSOR": XGBRegressor(),        # ✅ renamed
            "CATBOOST REGRESSOR": CatBoostRegressor(verbose=False), # ✅ renamed
            "ADABOOST REGRESSOR": AdaBoostRegressor()
        }

        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models
        )

        # ✅ Find best model by test_score
        best_model_name = max(model_report, key=lambda k: model_report[k]["test_score"])
        best_model_score = model_report[best_model_name]["test_score"]
        best_model = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No good model found (score < 0.6)")

        logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )

        predicted = best_model.predict(X_test)
        r2_square = r2_score(y_test, predicted)

        return r2_square

    except Exception as e:
        raise CustomException(e, sys)
