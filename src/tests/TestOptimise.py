from bayes_opt.util import Colours
from ..models.otimise import optimise_lightgbm, optimise_xgb
from ..data.make_dataset import get_data

if __name__ == "__main__":
    data, targets = get_data()

    print(Colours.yellow("--- Optimizing SVM ---"))
    optimise_lightgbm(data, targets)

    print(Colours.green("--- Optimizing Random Forest ---"))
    optimise_xgb(data, targets)