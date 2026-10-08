"""Optional Python 3.10 CPU smoke for the separate MetaWorld environment."""
import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args(); out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    data = dict(observations=np.zeros((2,8,39),dtype=np.float32),
                actions=np.zeros((2,8,4),dtype=np.float32),
                rewards=np.ones((2,8),dtype=np.float32), values=np.ones((2,8),dtype=np.float32))
    with (out/'data.pkl').open('wb') as f: pickle.dump(data,f)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled', OMP_NUM_THREADS='1')
    commands = []
    def run(script,*args):
        import json
        command = [sys.executable,script,*map(str,args)]
        result = subprocess.run(command,env=env,capture_output=True,text=True)
        (out/f'{len(commands)}_{script}.log').write_text(result.stdout+result.stderr)
        commands.append({'command':command,'returncode':result.returncode})
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        if result.returncode: print(result.stderr[-10000:]); raise RuntimeError(script)
        print('PASS',script,*map(str,args),flush=True)
    for script in ['train_ppo_ml1_all_goals.py','collect_metaworld_data.py','train_icl_metaworld.py','eval_icl_metaworld.py']:
        run(script,'--help')
    # Preserve policy normalization while storing raw pre-action observations.
    from stable_baselines3.common.vec_env import VecNormalize
    from collect_metaworld_data import stored_icl_observations
    from train_icl_metaworld import model_batch
    normalizer = object.__new__(VecNormalize)
    normalizer.old_obs = np.full((1,39), 2.5)
    policy_obs = np.zeros((1,39))
    snapshot = stored_icl_observations(normalizer, policy_obs, True)
    np.testing.assert_array_equal(snapshot, normalizer.old_obs)
    snapshot[:] = -1
    np.testing.assert_array_equal(normalizer.old_obs, 2.5)
    np.testing.assert_array_equal(policy_obs, 0)
    batch = {'context_rewards': np.zeros((1,8,1)), 'context_values': np.ones((1,8,1))}
    assert model_batch(batch, 'auto_relabel', True)['context_rewards'] is batch['context_values']
    assert model_batch(batch, 'auto_relabel', False)['context_rewards'] is batch['context_rewards']
    for label in ['ad','auto_relabel','cv_value_tokens']:
        method = 'auto_relabel' if label == 'cv_value_tokens' else label
        run('train_icl_metaworld.py','--train_data',out/'data.pkl','--algorithm',method,
            '--output-dir',out/label,'--horizon',8,'--n_embd',16,'--n_layer',1,'--n_head',1,
            '--num_epochs',1,'--batch_size',2,'--num_workers',0,
            *(['--value-token-input'] if label == 'cv_value_tokens' else []))
        for task in ['reach-v3','push-v3']:
            run('eval_icl_metaworld.py','--env-name',task,'--model_dir',out/label,'--epoch','final',
                '--algorithm',method,'--split','test','--num_tasks',1,'--num_steps',3,
                '--horizon',8,'--n_embd',16,'--n_layer',1,'--n_head',1,
                '--output-json',out/f'{label}_{task}.json')
    print('All optional MetaWorld smoke commands passed; these are synthetic checkpoints.',flush=True)


if __name__ == '__main__': main()
