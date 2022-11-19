import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import pickle

class Pipeline_diabetes:
    """Clase diseñada para crear un pipeline del dataset
    https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
    """
    def __init__(self, dataframe: pd.DataFrame, cols: list, seed: int):
        self.dataframe = dataframe
        self.cols = cols
        self.seed = seed

    def select_features(self, target: str) -> pd.DataFrame:
        """Funcion encargada de seleccionar los features y el target de un DataFRame

        :param target: Nombre del target del DataFRame
        :type target: str
        :return: Devuelve dos variables donde uno son los features y el otro es el target
        :rtype: pd.DataFrame
        """
        self.dataframe.columns = self.cols
        X = self.dataframe.drop(columns=target)
        y = self.dataframe.loc[:,target]
        return X, y

    def split_dataframe(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Funcion encargada de dividir los features y target en dos conjuntos de datos:
        entrenamiento y test

        :param X: Objeto DataFrame que contiene los features
        :type X: pd.DataFrame
        :param y: Objeto DataFrame que contiene el target
        :type y: pd.DataFrame
        :return: Conjunto de datos empleados para el entrenamiento y testeo
        :rtype: pd.DataFrame
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            random_state=self.seed
        )
        return X_train, X_test, y_train, y_test

    def build_pipeline(self, X: pd.DataFrame):
        """Funcion encargada de construir un pipeline listo para predecir

        :param X: Datos empleados para entrenar el pipeline
        :type X: pd.DataFrame
        :return: Pipeline entrenado
        :rtype: Pipeline sklearn
        """
        return make_pipeline(
                ColumnTransformer([
                    ("pipe_num",
                    make_pipeline(StandardScaler(),
                                KNNImputer())),
                    list(X.select_dtypes('number').columns)

                ]),
                RandomForestClassifier()
            )

    def export_pipeline(self, name:str, pipeline, mode:str="wb"):
        """Funcion encargada de generar un pickle del pipeline entrenado

        :param name: Nombre del pickle
        :type name: str
        :param pipeline: Pipeline sklearn
        :type pipeline: Pipeline sklearn
        :param mode: modo de escritura defaults to "wb"
        :type mode: str, optional
        """
        with open(f"{name}.pkl", mode) as f:
            pickle.dump(pipeline, f)

if __name__ == "__main__":

    from nivel2.pipeline_diabetes import Pipeline_diabetes

    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv')
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    diabetes = Pipeline_diabetes(dataframe=df,
                    cols=cols,
                    seed=42)

    X, y = diabetes.select_features(target='Outcome')

    X_train, X_test, y_train, y_test= diabetes.split_dataframe(X,y)

    pipeline = diabetes.build_pipeline(X_train)

    diabetes.export_pipeline('model', pipeline)