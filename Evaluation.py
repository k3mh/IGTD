### Evaluation
import numpy as np
def Precision(ds_meta, explanation, partial=False):
    PERC = 0
    tp = []
    length = 0
    try:
        if ds_meta.shape[0] != explanation.shape[0]:
            raise ValueError
    except ValueError:
        print("Explanations and metadata are not the same length")


    if partial:
        for row in ds_meta.index.to_list():
            print("row=", row)
            print(ds_meta.loc[row, "imp_vars"], explanation.set_index("instance").loc[row, "features"])
            intersection = len(set(ds_meta.loc[row, "imp_vars"]).intersection(set(explanation.set_index("instance").loc[row, "features"])))
            if intersection > 0:
                tp.append(1)
            print (tp)


        print("===============")
        print(tp)
        print(len(tp))
        if len(tp) == 0 :
            PERC = 0
        else:
            PERC = np.sum(tp) / len(tp)

    else:
        for row in ds_meta.index.to_list():
            print("row=", row)
            print(ds_meta.loc[row, "imp_vars"], explanation.set_index("instance").loc[row, "features"])

            #length += len(ds_meta.loc[row, "imp_vars"])
            intersection = len(set(ds_meta.loc[row, "imp_vars"]).intersection(set(explanation.set_index("instance").loc[row, "features"])))
            ex_vars_num =  len(set(ds_meta.loc[row, "imp_vars"]))
            tp.append(intersection /ex_vars_num)

            print(tp, length)
        PERC = np.sum(tp) / len(tp)

    return PERC, tp

def FPR(ds_meta, explanation, all_features):
    # FPR = FP/N
    FPR = 0
    FP = []
    length = 0
    try:
        if ds_meta.shape[0] != explanation.shape[0]:
            raise ValueError
    except ValueError:
        print("Explanations and metadata are not the same length")

    for row in ds_meta.index.to_list():

        #length += len(ds_meta.loc[row, "imp_vars"])
        fp = len(set(explanation.set_index("instance").loc[row, "features"]).difference(set(ds_meta.loc[row, "imp_vars"])))
        tn     = len(set(all_features).difference(set(explanation.set_index("instance").loc[row, "features"])))
        neg = tn + fp
        FP.append(fp / neg)

        # print("row=", row)
        # print(ds_meta.loc[row, "imp_vars"], explanation.set_index("instance").loc[row, "features"])
        # print(fp, neg)
        # print(FP)
        # print('---------------')

    FPR = np.sum(FP) / len(FP)

    return FPR, FP


def sensetivity(ds_meta, explanation, top_features=2):

    sens = []
    sens_all = 0
    for row in ds_meta.index.to_list():
        #print("row=", row)
        #print(ds_meta.loc[row, "imp_vars"], explanation.set_index("instance").loc[row, "features"])
        features = explanation.set_index("instance").loc[row, "features"][:top_features]
        importance = explanation.set_index("instance").loc[row, "importance"][:top_features]

        pred = [Y for X, Y in sorted(zip(np.abs(importance), features))][::-1]
        actual = ds_meta.loc[row, "imp_vars"]

        common_rank = np.sum([1 for x, y in zip(pred, actual) if x == y])
        sens.append(common_rank/len(actual))

    sens_all = np.sum(sens) / ds_meta.shape[0]
    return sens_all, sens

def  overall_score (eval_all, ds_complexity, accu_list):
    score_lst =[]
    for iter, ds in enumerate(eval_all.dataset.unique()):
        avg_score =  np.mean(eval_all.loc[(eval_all.dataset == ds) & (
            eval_all.metric.isin(["percision", "FPR", "sensitivity"])), 'score'])
        complexity = ds_complexity[iter]
        accu = accu_list[iter]
        score_lst.append(avg_score * complexity / accu)

    total_scr = np.sum(score_lst)
    return  (total_scr, score_lst)

