"""CPU smoke: tiny synthetic training, checkpoint reload, real short rollouts.

Run from the repository root: python tests/smoke_release.py --output-dir outputs/smoke
This validates execution and information flow; it does not reproduce paper scores.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--skip-help', action='store_true', help='Resume training smoke after separately validating CLI imports.')
    p.add_argument('--procgen', action='store_true', help='Linux Procgen extra must be installed.')
    a = p.parse_args()
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    rng = np.random.RandomState(5)
    data = dict(observations=rng.randint(0, 8, (4,8,3,7,7)).astype(np.uint8),
                actions=rng.randint(0,7,(4,8)).astype(np.int64),
                rewards=rng.uniform(0,.8,(4,8)).astype(np.float32),
                dones=np.tile(np.array([0,0,0,1,0,0,0,1],bool),(4,1)))
    for split in ('train','val'):
        with (out/f'{split}.pkl').open('wb') as f: pickle.dump(data,f)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled',
                                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', SDL_VIDEODRIVER='dummy')
    commands = []
    def run(script, *args):
        command = [sys.executable, script, *map(str,args)]
        print('RUN', ' '.join(command), flush=True)
        result = subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        commands.append({'command':command,'returncode':result.returncode})
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        (out/(script.replace('.py','')+f'_{len(commands)}.log')).write_text(result.stdout)
        if result.returncode: print(result.stdout[-12000:]); raise RuntimeError(f'{script} failed')
        print('PASS', script, flush=True)
    scripts=['train_cv_minigrid.py','eval_cv_minigrid.py','build_ppo_state_targets.py',
             'train_reinformer_minigrid.py','eval_reinformer_minigrid.py',
             'train_ic_cql_minigrid.py','eval_ic_cql_minigrid.py',
             'train_icl_minigrid.py','eval_icl_minigrid.py','train_icl_darkroom.py','eval_icl_darkroom.py',
             'collect_minigrid_data.py','collect_darkroom_data.py']
    if a.procgen: scripts += ['train_icl_procgen.py','eval_icl_procgen.py','train_procgen_ppo_official.py','collect_procgen_official.py']
    if not a.skip_help:
        for script in scripts: run(script,'--help')
    for method in ('cv','ad'):
        run('train_cv_minigrid.py','--env','MiniGrid-SimpleCrossingS9N3-v0',
            '--train-data',out/'train.pkl','--val-data',out/'val.pkl','--output-dir',out/method,
            '--method',method,'--H',8,'--embd',16,'--layer',1,'--head',1,'--num-epochs',1,
            '--batch-size',2,'--max-batches',1,'--device','cpu')
        conditions = ('normal','zero','frozen') if method == 'cv' else ('normal',)
        for condition in conditions:
            run('eval_cv_minigrid.py','--checkpoint',out/method/'epoch_1.pt',
                '--eval-env','MiniGrid-SimpleCrossingS9N3-v0','--output-json',out/f'{method}_{condition}.json',
                '--method',method,'--condition',condition,'--num-envs',1,'--num-steps',12,'--device','cpu')
    for method in ('reinformer','ic_cql'):
        run(f'train_{method}_minigrid.py','--env','MiniGrid-SimpleCrossingS9N3-v0',
            '--train-data',out/'train.pkl','--val-data',out/'val.pkl','--output-dir',out/method,
            '--train-histories-per-stream',1,'--val-histories-per-stream',1,
            '--H',8,'--embd',16,'--layer',1,'--head',1,'--num-epochs',1,
            '--batch-size',2,'--grad-accum-steps',1,'--num-workers',0,
            '--max-train-batches',1,'--max-val-batches',1,'--device','cpu',
            *(['--amp-dtype','none','--warmup-updates','1'] if method=='reinformer' else []))
        run(f'eval_{method}_minigrid.py','--checkpoint',out/method/'epoch_1.pt',
            '--eval-env','MiniGrid-SimpleCrossingS9N3-v0','--output-json',out/f'{method}_eval.json',
            '--num-envs',1,'--num-steps',12,'--device','cpu')
    if a.procgen:
        proc = dict(observations=rng.randint(0,255,(2,8,3,64,64)).astype(np.uint8),
                    actions=rng.randint(0,15,(2,8)).astype(np.int64),
                    rewards=rng.uniform(0,1,(2,8)).astype(np.float32),
                    values=rng.uniform(0,1,(2,8)).astype(np.float32),dones=np.zeros((2,8),dtype=bool))
        np.savez(out/'procgen.npz',**proc)
        for method in ('ad','cv'):
            run('train_icl_procgen.py','--env','bigfish','--H',4,'--embd',16,'--layer',1,'--head',1,
                '--num_epochs',1,'--batch-size',2,'--num-workers',0,'--device','cpu',
                '--max_batches_per_epoch',1,'--train-data',out/'procgen.npz','--val-data',out/'procgen.npz',
                '--output-dir',out/f'procgen_{method}',*(['--auto_relabel'] if method=='cv' else []))
            ckpt = 'epoch1_auto_relabel.pt' if method=='cv' else 'epoch1_more.pt'
            run('eval_icl_procgen.py','--env','bigfish','--eval_env','bigfish','--H',4,
                '--embd',16,'--layer',1,'--head',1,'--num_envs',1,'--num_steps',6,
                '--checkpoint',out/f'procgen_{method}'/ckpt,'--output-json',out/f'procgen_{method}_eval.json',
                *(['--auto_relabel'] if method=='cv' else []))
    print(f'All {len(commands)} commands passed. Synthetic scores are not paper results.',flush=True)


if __name__ == '__main__': main()
