import pickle
import warnings
from matplotlib import pyplot as plt
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import reciprocal_rank


warnings.filterwarnings('ignore')


MODEL_PATH = "./models/model_ckpt.pkl"
BENCHMARK_DATA_PATH = "./benchmark/data/eval_data.pkl"
VIS_PATH = "./reports/figures"
BENCH_EVAL_VIS_PATH = VIS_PATH + "/bench_eval_vis.png"


with open(MODEL_PATH, "rb") as file:
    model_data = pickle.load(file)
user_features = model_data["user_features"]
item_features = model_data["item_features"]
model = model_data["model"]
with open(BENCHMARK_DATA_PATH, "rb") as file:
    bench_data = pickle.load(file)
test_interactions = bench_data["test_interactions"]
print("Benchmark AUC score:",
      auc_score(model, test_interactions, user_features=user_features,
           item_features=item_features).mean(),
      sep="\t"
)
print("Benchmark Reciprocal rank:",
      reciprocal_rank(model, test_interactions, user_features=user_features,
                      item_features=item_features).mean(),
      sep="\t")
test_prec_y = []
test_rec_y = []
ks = list(range(1, 11))
for k in ks:
    test_prec_y.append(
        precision_at_k(model, test_interactions, k=k,
                       user_features=user_features,
                       item_features=item_features).mean()
    )
    test_rec_y.append(
        recall_at_k(model, test_interactions, k=k,
                    user_features=user_features,
                    item_features=item_features).mean()
    )
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].plot(ks, test_prec_y, marker="o", label="Test")
axes[0].title.set_text("Benchmark Precision at K of the LightFM")
axes[0].set_ylabel("Precision")
axes[0].set_xlabel("K")
axes[1].plot(ks, test_rec_y, marker="o", label="Test")
axes[1].title.set_text("Benchmark Recall at K of the LightFM")
axes[1].set_ylabel("Recall")
axes[1].set_xlabel("K")
fig.savefig(BENCH_EVAL_VIS_PATH)
