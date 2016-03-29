import PredictiveModels as pm
from RationaleSelect import *
import matplotlib.pyplot as plt


def gen_curve(data,target,step):
    indices = np.random.shuffle(np.arange(len(target)))
    mae_curve = []
    sublens = np.arange(step, len(target), step)
    for sublen in sublens:
        sub = indices[sublen]
        mae = pm.get_ridge_cv_results(np.column_stack((data[sub],target[sub])), 45, 10)[0]
        mae_curve.append(mae)
    plt.plot(sublens, mae_curve, 'ro')
    plt.show()

userid = 28451
user_ratings = get_matched_user_ratings(userid)
movies = user_ratings.keys()
ratings = user_ratings.values()

id_dict = get_id_row_dict()
idxs = [id_dict[k] for k in movies]
avg_and_num = sort_avg()

genre_matrix = get_genre_matrix()[idxs]
user_key_matrix  = get_key_matrix()[idxs]
actor_matrix = get_actor_matrix()[idxs]

pp.scale(avg_and_num[:,0], with_mean=True, with_std=True, copy=False)
pp.scale(avg_and_num[:,1], with_mean=True, with_std=True, copy=False)
data = np.column_stack((avg_and_num[idxs], genre_matrix, user_key_matrix, actor_matrix))
target = ratings

step = 10
gen_curve(data,target,step)