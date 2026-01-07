from vllm import SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval.vllm_eval import VLLM_Eval
from cs336_alignment.utils import load_math_dataset

vllm = VLLM_Eval(
    model_id="Qwen/Qwen2.5-Math-1.5B",
    device='cuda:0',
    seed=0,
)
all_test = load_math_dataset(sample=100)

sampling_params = SamplingParams(
    temperature=1.0, top_p=0.9, max_tokens=1024
)

metrics = vllm.evaluate_vllm(
    prompts=all_test['prompt'].tolist(),
    eval_sampling_params=sampling_params,
    answers=all_test['solution'].tolist(),
    reward_fn=r1_zero_reward_fn,
    print_metrics=True,
)
import pdb; pdb.set_trace()
