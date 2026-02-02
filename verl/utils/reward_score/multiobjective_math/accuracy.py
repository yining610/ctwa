from verl.utils.reward_score.math_reward import compute_score as compute_accuracy_boxed

def compute_score(solution_str: str, ground_truth: str, extra_info: dict) -> float:

    return compute_accuracy_boxed(solution_str, ground_truth)