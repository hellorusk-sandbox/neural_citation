from ncn.evaluation import Evaluator
from ncn.data import get_datasets
import numpy as np

path_to_weights = "./models/NCN_11_2_12_embed_128_hid_256_1_GRU.pt"
path_to_data = "./data/arxiv_data.csv"
data = get_datasets(path_to_data, 20000, 20000, 20000)

evaluator = Evaluator([4,4,5,6,7], [1,2], 256, 128, 1, path_to_weights, data, evaluate=True, show_attention=False)

at_10 = evaluator.recall(10)

print(round(at_10, 4))