from verl.utils.reward_score.math_reward import compute_score as compute_accuracy_boxed

# def compute_score(solution_str, ground_truth, extra_info) -> float:

#     # Gate conciseness by correctness to prevent reward hacking:
#     # otherwise the easiest way to get conciseness=1 is to answer very short but wrong.
#     correct = float(compute_accuracy_boxed(solution_str, ground_truth))
#     if correct < 1.0:
#         return 0.0

#     if "global_avg_length" not in extra_info:
#         # at the very beginning of training.
#         return 1.0

#     response_length = extra_info["response_length"]
#     global_avg_length = extra_info["global_avg_length"]

#     if response_length >= global_avg_length:
#         return 0.0
#     else:
#         return 1.0


def compute_score(solution_str, ground_truth, extra_info) -> float:

    if "global_avg_length" not in extra_info:
        # at the very beginning of training.
        return 1.0

    response_length = extra_info["response_length"]
    global_avg_length = extra_info["global_avg_length"]

    if response_length >= global_avg_length:
        return 0.0
    else:
        return 1.0