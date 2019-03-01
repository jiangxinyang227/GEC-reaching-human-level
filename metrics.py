def get_corr_pred(pred, true):
    count = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            count += 1

    return count


def f_beta(y_pred, y_true, sequence_len, beta=0.5):
    y_pred = y_pred.tolist()
    y_true = y_true.tolist()
    sequence_len = sequence_len.tolist()

    f_bs = []
    for i in range(len(y_pred)):
        correct_preds = get_corr_pred(y_pred[i][:sequence_len[i]], y_true[i][:sequence_len[i]])
        all_preds = len(y_pred[i][:sequence_len[i]])
        all_trues = len(y_true[i][:sequence_len[i]])
        precision = correct_preds / all_preds if correct_preds > 0 else 0
        recall = correct_preds / all_trues if correct_preds > 0 else 0

        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall) if correct_preds > 0 else 0

        f_bs.append(f_b)

    return sum(f_bs) / len(f_bs)


