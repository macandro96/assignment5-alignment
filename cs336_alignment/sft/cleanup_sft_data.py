import pandas as pd

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

if __name__ == "__main__":
    df = pd.read_json("data/math/sft.jsonl", lines=True)
    
    def find_think_trace(x):
        if "</think>" not in x:
            return None
        else:
            return x.split("</think")[0].strip()
    
    def find_answer(x):
        if "<answer>" not in x or "</answer>" not in x:
            return None
        else:
            return x.split("<answer>")[1].split("</answer>")[0].strip()
    
    df['think_trace'] = df['response'].apply(lambda x: find_think_trace(x))
    df['answer_trace'] = df['response'].apply(lambda x: find_answer(x))
    
    def combine_think_answer(think_trace, answer_trace):
        if think_trace and answer_trace:
            return f"{think_trace}</think> <answer>{answer_trace}</answer>"

    df['think_answer'] = df.apply(lambda x: combine_think_answer(x['think_trace'], x['answer_trace']), axis=1)
    
    def compute_reward(pred, gt):
        if pred and gt:
            return r1_zero_reward_fn(pred, gt)
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}
    
    df['question'] = df['prompt'].apply(lambda x: x.split("User:")[1].split("Assistant")[0].strip())
    
    train_set = pd.read_json("data/math/train.jsonl", lines=True)
    merged = pd.merge(df, train_set, left_on='question', right_on='problem', how='left')
    merged['reward'] = merged.apply(lambda x: compute_reward(x['think_answer'], x['solution']), axis=1)
    merged['format_reward'] = merged['reward'].apply(lambda x: x['format_reward'])
    merged['answer_reward'] = merged['reward'].apply(lambda x: x['answer_reward'])

    merged = merged[['prompt', 'think_answer', 'format_reward', 'answer_reward', 'reward']]
    merged = merged.rename(columns={'think_answer': 'response'})
    merged = merged[merged['response'].notna()]
    merged.to_json('data/math/sft_clean.jsonl', lines=True, orient='records')
    # merged['format_reward'] = merged['reward'].apply(lambda x: x['format_reward'])
    # df['reward'] = df.apply(lambda x: compute_reward(x['think_answer']))