
def f_beta(y_pred, y_true, beta=0.5):

    f_bs = []
    for i in range(len(y_pred)):
        correct_preds = len(set(y_pred[i]) & set(y_true[i]))
        all_preds = len(y_pred[i])
        all_trues = len(y_true[i])
        precision = correct_preds / all_preds if correct_preds > 0 else 0
        recall = correct_preds / all_trues if correct_preds > 0 else 0

        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall) if correct_preds > 0 else 0

        f_bs.append(f_b)

    return sum(f_bs) / len(f_bs)