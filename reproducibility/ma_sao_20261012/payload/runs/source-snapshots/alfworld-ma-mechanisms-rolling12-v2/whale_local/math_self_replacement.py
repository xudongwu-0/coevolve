"""One-cycle replacement after an audited, dual-gain external WHALE baseline.

The original W0->W1 RSFT prefix is reused and charged equally. Only the
internal proposer is new; its code tokens receive real outcome-GRPO updates.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import pandas as pd

from alternating_common import check_proposals, requests_for, run_editor, source_snapshot
from alternating_cycles import explored_proposals
from alternating_pool import Pool
from harness_bootstrap import START
from harness_focused_bootstrap import choose_focus
from meta_harness.benchmark import compute_pareto_frontier
from meta_harness.meta_harness_retool import pick_accepted_from_frontier
from pipeline import ROOT, sha, write_json


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump_rows(p, rs):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r) + '\n' for r in rs))


def status(run, stage, **extra):
    v = {'stage': stage, 'utc': utc(), **extra}
    write_json(run / 'STATUS.json', v); print(json.dumps(v), flush=True)


def model_spec(model, harness, data, output, schedule=None):
    result = {**model, 'harness': str(harness), 'data': data, 'output': str(output),
        'draws': 1, 'seed': 42, 'temperature': 1.0}
    if schedule is not None: result['schedule'] = schedule
    return result


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'runs/whale_math_mean8_native_20260915'
    cfg = read(prior / 'config.json'); audit = read(prior / 'verification.json')
    assert audit['passed_audit'] and audit['qualified_for_gate']
    first_draw = {}
    for key in ['W0_H0', 'W1_H0', 'W0_Hf', 'W1_Hf']:
        p = prior / 'confirmation' / key / 'trajectories.jsonl'
        selected = [r for r in rows(p) if r['draw'] == 0]
        assert len(selected) == 128
        first_draw[key] = {'solved': int(sum(r['reward'] for r in selected)), 'total': len(selected),
            'source': str(p), 'sha256': sha(p), 'draw': 0}
    assert first_draw['W1_Hf']['solved'] - first_draw['W0_Hf']['solved'] >= 7
    assert first_draw['W1_Hf']['solved'] - first_draw['W1_H0']['solved'] >= 7
    write_json(run / 'BASELINE_QUALIFICATION.json', {'passed': True, 'first_draw': first_draw,
        'source_audit': str(prior / 'verification.json'), 'source_audit_sha256': sha(prior / 'verification.json'),
        'new_evaluation_draws': 0, 'selection_rule': 'Reuse draw zero, never select a favorable draw.',
        'historical_exposure': True})
    (run / 'initial_harness.py').write_text(START)
    external = run / 'external_harness.py'
    shutil.copy2(read(prior / 'frozen_final.json')['harness_path'], external)
    w1 = prior / 'round_01/checkpoint'
    editor = ROOT / 'runs/whale_dual2b_focused_20260916/editor_continuation/checkpoint'
    models = {'W0': {'model': cfg['base_model'], 'weight_digest': read(prior / 'initial_identity.json')['weight_digest']},
        'W1': {'model': str(w1), 'weight_digest': read(w1 / 'training_state.json')['weight_digest']}}
    config = {'reference': str(prior), 'models': models, 'solver_initial_checkpoint': str(w1),
        'editor_initial_checkpoint': str(editor), 'editor_initial_digest': read(editor / 'training_state.json')['weight_digest'],
        'initial_harness': str(run / 'initial_harness.py'), 'external_harness': str(external),
        'initial_harness_sha256': sha(run / 'initial_harness.py'), 'external_harness_sha256': sha(external),
        'data': {'splits': {k: cfg['data']['splits'][k] for k in ['train', 'mh_val', 'test']}},
        'pool_gpus': ['7', '8', '9'] + [f'node1:{i}' for i in range(10)],
        'checkpoint_allowlist': [m['model'] for m in models.values()],
        'editor_gpu': '6', 'pre_proposals': 8, 'post_proposals': 4, 'selection_candidates': 2,
        'editor_updates': 4, 'proposal_temperature': 1.0, 'reward_task_count': 8,
        'seed': 42, 'evaluation_draws': 1, 'harness_rollout_cap': 240,
        'shared_rsft_prefix': {'trajectories_per_arm': 512, 'optimizer_updates': 120,
            'new_updates': 0, 'training_state_sha256': sha(w1 / 'training_state.json')},
        'budget_allocation': '48 incumbent + 8 proposals * 8 + 4 post-update proposals * 8 + 2 shortlisted * 48 = 240 maximum. Cached task results are reused and counted once.',
        'reward_batch_selection': 'First four failures then first four successes of incumbent dev48, fill remaining slots in original index order; no test feedback.',
        'invalid_reward': 'zero success minus incumbent batch success; no-op has delta zero',
        'selection': 'Only post-update valid candidates; native Pareto accuracy/turn selection including incumbent.',
        'limitations': ['Small MATH number_theory Level2-4 integer subset, not full paper DAPO/AIME.',
            'Test128 has historical use, but is not opened by this run until all candidates/checkpoints are frozen.',
            'External baseline reuses audited full-code native MH; internal uses outcome-GRPO and a restricted code grammar.',
            'Both can change prompts, partial-response recovery and error-feedback hooks; internal can emit only the supported class grammar.',
            '2048 assistant tokens / 1024 per call / 4096 context; no thinking; up to three harness turns.',
            'Equal target-trajectory ceiling, not equal external FLOPs or wall time. Internal SFT initialization is an extra historical cost.']}
    write_json(run / 'config.json', config)
    write_json(run / 'PROTOCOL.json', {'registered_utc': utc(), 'config_sha256': sha(run / 'config.json'),
        'new_final_evaluations_max': 6 * 128, 'current_test_used_for_selection': False,
        'phase_order': 'Audited W0/H0 -> native RSFT W1/H0 -> external or internal H proposal and selection',
        'editor_update_rule': 'One group of eight, four outcome-GRPO updates; skip if zero reward variance.',
        'no_extra_proposals_after_results': True, 'no_multi_seed': True})
    source_snapshot(run, 'frozen', ['math_self_replacement.py', 'alternating_cycles.py'])
    status(run, 'prepared')


def summarize(output, data, model, harness, records):
    records = sorted(records, key=lambda r: (r['draw'], r['example_index']))
    assert len({(r['draw'], r['example_index']) for r in records}) == len(records)
    dump_rows(output / 'trajectories.jsonl', records)
    value = {'num_examples': len(records), 'solved_examples': int(sum(r['reward'] for r in records)),
        'success_rate': sum(r['reward'] for r in records) / len(records),
        'mean_turn_count': sum(len(r['policy_calls']) for r in records) / len(records),
        'metadata': {'weight_digest': model['weight_digest'], 'harness_sha256': sha(harness),
            'dataset_path': data['path'], 'dataset_sha256': data['sha256'],
            'sampling_schedule': [[r['draw'], r['example_index']] for r in records],
            'temperature': 1.0, 'seed': 42, 'assistant_token_budget': 2048, 'policy_max_tokens': 1024}}
    write_json(output / 'val.json', value); return value


def score(pool, run, cfg, directory, indices, baseline):
    generated = rows(directory / 'generations.jsonl')
    checks = check_proposals(directory, cfg['data']['splits']['mh_val'])
    baseline_score = sum(r['reward'] for r in rows(baseline / 'trajectories.jsonl')) / len(indices)
    mapped = {}; jobs = {}; model = cfg['models']['W1']; data = cfg['data']['splits']['mh_val']
    for g in generated:
        if not g['valid_source'] or not checks[g['id']]['passed']:
            mapped[g['id']] = ('invalid', None); continue
        path = directory / (g['id'] + '.py')
        if g['source'].strip() == Path(cfg['initial_harness']).read_text().strip():
            mapped[g['id']] = ('no_op', None); continue
        output = run / 'outcome_cache' / sha(path)
        mapped[g['id']] = ('valid', output)
        jobs[str(output)] = model_spec(model, path, data, output, [[0, i] for i in indices])
    pool.evaluate_many(list(jobs.values()))
    scored = []
    for g in generated:
        kind, output = mapped[g['id']]
        success = read(output / 'val.json')['success_rate'] if output else (baseline_score if kind == 'no_op' else 0.0)
        scored.append({**g, 'outcome_kind': kind, 'reward_delta': success - baseline_score,
            'reward_provenance': {'candidate': str(output) if output else None, 'baseline': str(baseline),
                'candidate_score': success, 'baseline_score': baseline_score,
                'invalid_receives_zero_success': kind == 'invalid', 'indices': indices}})
    dump_rows(directory / 'scored.jsonl', scored)
    return scored


def execute(run):
    cfg = read(run / 'config.json'); assert sha(run / 'config.json') == read(run / 'PROTOCOL.json')['config_sha256']
    started = time.monotonic()
    if not (run / 'START.json').exists(): write_json(run / 'START.json', {'utc': utc()})
    else: write_json(run / 'RESUME.json', {'utc': utc(), 'reason': 'User requested a smaller rapid-iteration evaluation.'})
    h0 = Path(cfg['initial_harness']); model = cfg['models']['W1']; data = cfg['data']['splits']['mh_val']
    editor = Path(cfg['editor_initial_checkpoint'])
    try:
        with Pool(run / 'pool', cfg) as pool:
            status(run, 'incumbent_development')
            incumbent = run / 'incumbent'
            pool.evaluate_many([model_spec(model, h0, data, incumbent)])
            base = rows(incumbent / 'trajectories.jsonl')
            selected = [r for r in base if not r['reward']][:4] + [r for r in base if r['reward']][:4]
            ids = {r['example_index'] for r in selected}
            selected += [r for r in base if r['example_index'] not in ids][:8 - len(selected)]
            indices = sorted(r['example_index'] for r in selected); assert len(indices) == 8
            reward_base = run / 'reward_baseline'
            summarize(reward_base, data, model, h0, selected)
            focus = choose_focus(base, h0.read_text(), 0)
            requests = requests_for('reward_0', h0.read_text(), reward_base, focus, 8, 94200)
            write_json(run / 'REWARD_BATCH.json', {'indices': indices, 'focus': focus,
                'incumbent_development_sha256': sha(incumbent / 'trajectories.jsonl')})
            status(run, 'internal_pre_generation', focus=focus)
            pre = run / 'pre_proposals'
            explored_proposals(editor, pre, requests, cfg['proposal_temperature'])
            scored = score(pool, run, cfg, pre, indices, reward_base)
            status(run, 'editor_outcome_grpo', rewards=[r['reward_delta'] for r in scored])
            trained = run_editor(run, editor, run / 'editor_grpo', scored, mode='train', epochs=4)
            if trained['updated']: editor = Path(trained['identity']['weights'])
            status(run, 'internal_post_generation', updated=trained['updated'])
            post = run / 'post_proposals'
            explored_proposals(editor, post, requests[:4], cfg['proposal_temperature'])
            after = score(pool, run, cfg, post, indices, reward_base)
            unique = {}
            for r in sorted(after, key=lambda r: (-r['reward_delta'], r['id'])):
                if r['outcome_kind'] == 'valid': unique.setdefault(hashlib.sha256(r['source'].encode()).hexdigest(), r)
            shortlist = list(unique.values())[:2]
            status(run, 'native_harness_selection', candidate_ids=[r['id'] for r in shortlist])
            choices = {'incumbent': (h0, incumbent)}
            missing = [[0, i] for i in range(48) if i not in indices]
            for r in shortlist:
                path = post / (r['id'] + '.py'); output = run / 'selection' / r['id']
                supplement = output / 'remaining'
                pool.evaluate_many([model_spec(model, path, data, supplement, missing)])
                records = rows(Path(r['reward_provenance']['candidate']) / 'trajectories.jsonl') + rows(supplement / 'trajectories.jsonl')
                summarize(output, data, model, path, records)
                choices[r['id']] = (path, output)
            values = {k: read(p / 'val.json') for k, (_, p) in choices.items()}
            selection = run / 'selection'; selection.mkdir(exist_ok=True)
            for name, (path, _) in choices.items():
                dest = selection / 'harnesses' / name / 'harness.py'; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            frontier = {'_pareto': [{'harness': n, 'avg_success_rate': s, 'avg_mean_turn_count': t}
                for n, s, t in compute_pareto_frontier([(n, v['success_rate'], v['mean_turn_count']) for n, v in values.items()])]}
            _, accepted = pick_accepted_from_frontier(frontier, selection); assert accepted
            hint = selection / 'harnesses' / accepted / 'harness.py'
            write_json(selection / 'decision.json', {'accepted': accepted, 'frontier': frontier, 'results': values,
                'selected': str(hint), 'sha256': sha(hint), 'source_paths': {k: str(v[1]) for k, v in choices.items()}})
            write_json(run / 'FROZEN.json', {'utc': utc(), 'editor_updated': trained['updated'],
                'editor_checkpoint': str(editor), 'editor_digest': trained['identity']['weight_digest'],
                'internal_harness': str(hint), 'internal_harness_sha256': sha(hint),
                'test_opened_in_this_run': False, 'search_elapsed_seconds': time.monotonic() - started})
            cache = {}; states = {}
            weight_names = ['W1'] if cfg.get('rapid_iteration') else ['W0', 'W1']
            harness_choices = [('H_internal', hint)] if cfg.get('rapid_iteration') else [
                ('H0', h0), ('H_external', Path(cfg['external_harness'])), ('H_internal', hint)]
            for wn in weight_names:
                for hn, hp in harness_choices:
                    name = wn + '_' + hn; m = cfg['models'][wn]; key = (m['weight_digest'], sha(hp))
                    if key not in cache:
                        output = run / 'test' / name
                        status(run, 'single_draw_test', state=name)
                        pool.evaluate_many([model_spec(m, hp, cfg['data']['splits']['test'], output)])
                        cache[key] = str(output)
                    states[name] = {'model': m, 'harness': str(hp), 'output': cache[key]}
                    write_json(run / 'TEST_STATES.json', states)
        write_json(run / 'COMPLETE.json', {'utc': utc(), 'elapsed_seconds': time.monotonic() - started})
        status(run, 'complete_pending_audit')
    except BaseException as exc:
        status(run, 'failed', error=repr(exc)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'run']); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); (prepare if args.action == 'prepare' else execute)(args.run.resolve())
