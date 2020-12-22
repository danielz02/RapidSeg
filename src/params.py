NUM_CLASS = 3
COLOR_MAP = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (1.0, 0.82745098039, 0.0, 1.0),
    5: (0.14901960784, 0.43921568628, 0.0, 1.0)
}
CLASS_NAME = {
    0: "Others",
    1: "Corn",
    5: "Soybeans"
}

data_path = "../data"
test_path = "../data/test"
mask_path = "../data/mask"
train_path = "../data/train"
mask_file = "CDL_2013_Champaign_north.tif"
test_file = "20130824_RE3_3A_Analytic_Champaign_south.tif"
train_file = "20130824_RE3_3A_Analytic_Champaign_north.tif"

train_mean = [5188.78934685, 4132.74818647, 2498.34747813, 3689.04702811, 11074.86217434]
train_std = [1482.89729162, 1447.21062441, 1384.91231294, 1149.82168184, 2879.24827197]

BATCH_SIZE = 32
