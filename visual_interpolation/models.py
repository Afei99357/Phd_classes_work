##import random forest regressor
from sklearn import ensemble, neighbors
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import image_io

if __name__ == "__main__":
    model = xgb.XGBRFRegressor(tree_method="hist", max_depth=30, n_jobs=-1)

    (
        (train_xy, train_rgb),
        (test_xy, test_rgb),
        (shape, all_xy, all_rgb)
    ) = image_io.image_to_train_test_all()

    model.fit(train_xy, train_rgb)
    pred_test_rgb = model.predict(test_xy)
    pred_rmse = mean_squared_error(pred_test_rgb, test_rgb)**0.5
    print(f"RMSE: {pred_rmse}")

    pred_all_rgb = model.predict(all_xy)
    image_io.predited_framelike_to_image(shape, pred_all_rgb, save_as="output2.webp", show=True)
    