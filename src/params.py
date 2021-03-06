NUM_CLASS = 3
COLOR_MAP = {
    # Comment out because of stupid sklearn "feature" in choosing color when background is designated manually
    # https://github.com/scikit-image/scikit-image/issues/2674
    # "Others": (0.0, 0.0, 0.0, 0.0),
    "Corn": (1.0, 0.82745098039, 0.0, 1.0),  # Yellow-ish
    "Soybeans": (0.14901960784, 0.43921568628, 0.0, 1.0)  # Green-ish
}
CLASS_NAME = {
    0: "Others",
    1: "Corn",
    2: "Soybeans"  # Soybean was 5 in the original class label
}

data_path = "../data"
test_path = "../data/test"
train_path = "../data/train"
model_path = "../saved_models"
mask_path = "../data/train_mask"
mask_file = "CDL_2013_Champaign_north.tif"
test_file = "20130824_RE3_3A_Analytic_Champaign_south.tif"
train_file = "20130824_RE3_3A_Analytic_Champaign_north.tif"

train_mean_rgb = [5188.78934685, 4132.74818647, 2498.34747813]
train_std_rgb = [1482.89729162, 1447.21062441, 1384.91231294]

train_mean_full = [5188.78934685, 4132.74818647, 2498.34747813, 3689.04702811, 11074.86217434]
train_std_full = [1482.89729162, 1447.21062441, 1384.91231294, 1149.82168184, 2879.24827197]

EPOCH = 60
BATCH_SIZE = 16
