### Evaluation
import numpy as np
import logging

logger = logging.getLogger()


def Recall(ds_meta, explanation, partial=False):
    logger.info("recall function start")

    RECL = 0
    tp = []
    length = 0
    try:
        if ds_meta.shape[0] != explanation.shape[0]:
            raise ValueError
    except ValueError:
        logger.info("Explanations and metadata are not the same length")


    if partial:
        for row in ds_meta.index.to_list():
            logger.debug(f"row={row}")
            logger.debug(f'{ds_meta.loc[row, "imp_vars"]}, {explanation.set_index("instance").loc[row, "features"]}')
            intersection = len(set(ds_meta.loc[row, "imp_vars"]).intersection(set(explanation.set_index("instance").loc[row, "features"])))
            if intersection > 0:
                tp.append(1)
            logger.debug (tp)


        logger.debug("===============")
        logger.debug(tp)
        logger.debug(len(tp))
        if len(tp) == 0 :
            RECL = 0
        else:
            RECL = np.sum(tp) / len(tp)

    else:
        for row in ds_meta.index.to_list():
            logger.debug(f"row= {row}")
            logger.debug('{ds_meta.loc[row, "imp_vars"]}, {explanation.set_index("instance").loc[row, "features"]}')

            #length += len(ds_meta.loc[row, "imp_vars"])
            intersection = len(set(ds_meta.loc[row, "imp_vars"]).intersection(set(explanation.set_index("instance").loc[row, "features"])))
            ex_vars_num =  len(set(ds_meta.loc[row, "imp_vars"]))
            tp.append(intersection /ex_vars_num)

            logger.debug(f"tp={tp}, length={length}")
        RECL = np.sum(tp) / len(tp)
        logger.info("Recal function end")


    return RECL, tp

def FPR(ds_meta, explanation, all_features):
    logger.info("FPR function start")

    # FPR = fp/N
    FPR = []
    try:
        if ds_meta.shape[0] != explanation.shape[0]:
            raise ValueError
    except ValueError:
        logger.info("Explanations and metadata are not the same length")

    for row in ds_meta.index.to_list():

        #length += len(ds_meta.loc[row, "imp_vars"])
        fp = len(set(explanation.set_index("instance").loc[row, "features"]).difference(set(ds_meta.loc[row, "imp_vars"])))
        # tn     = len(set(all_features).difference(set(explanation.set_index("instance").loc[row, "features"])))
        # neg = tn + fp
        ground_truth_neg = len(set(all_features).difference(set(ds_meta.loc[row, "imp_vars"])))


        FPR.append(fp/ ground_truth_neg)
        #
        # print("fp")
        # print(fp)
        # print("ground_truth_neg")
        # print(ground_truth_neg)
        # print("FPR")
        # print(FPR)


        # print("row=", row)
        # print(ds_meta.loc[row, "imp_vars"], explanation.set_index("instance").loc[row, "features"])
        # print(fp, neg)
        # print(FP)
        # print('---------------')

    avg_FPR = np.sum(FPR) / len(FPR)
    logger.info("FPR function end")


    return avg_FPR, FPR


def sensetivity(ds_meta, explanation, top_features=2):
    logger.info("sensetivity function start")
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
    logger.info("sensetivity function end")

    return sens_all, sens

def  overall_score (eval_all, ds_complexity, accu_list):
    logger.info("Overall function start")

    score_lst =[]
    for iter, ds in enumerate(eval_all.dataset.unique()):
        avg_score =  np.mean(eval_all.loc[(eval_all.dataset == ds) & (
            eval_all.metric.isin(["recall", "FPR", "sensitivity"])), 'score'])
        complexity = ds_complexity[iter]
        accu = accu_list[iter]
        score_lst.append(avg_score * complexity / accu)

    total_scr = np.sum(score_lst)
    return  (total_scr, score_lst)

