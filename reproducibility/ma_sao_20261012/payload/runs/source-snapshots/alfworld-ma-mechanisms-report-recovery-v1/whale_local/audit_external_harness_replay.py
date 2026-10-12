"""Verify frozen-model external search and report paired single-draw replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alternating_pool import read, rows
from external_harness_replay import without_system
from math_context_audit import recheck_context
from math_reward import binary_reward
from pipeline import sha, write_json


def audit(run):
    config, protocol = read(run / 'config.json'), read(run / 'PROTOCOL.json')
    assert sha(run / 'config.json') == protocol['config_sha256']
    reference, expanded = Path(config['reference']), Path(config['expanded_reference'])
    for root, key in [(reference, 'reference_audit_sha256'), (expanded, 'expanded_audit_sha256')]:
        assert sha(root / 'verification.json') == config[key]
        assert read(root / 'verification.json')['passed_audit']
    remote = read(run / 'REMOTE_RUNTIME_VERIFICATION.json')
    assert remote['passed'] and remote['same_runtime_as_previously_audited']
    assert sha(remote['prior_verification']) == remote['prior_sha256']
    prior_remote = read(remote['prior_verification'])
    assert remote['remote_runtime_hashes'] == prior_remote['remote_runtime_hashes']
    assert remote['local_runtime_hashes'] == prior_remote['local_runtime_hashes']
    for manifest in (run / 'source_snapshots').glob('*/manifest.json'):
        for path, digest in read(manifest).items():
            assert sha(path) == sha(manifest.parent / Path(path).name) == digest
    assert config['training_updates'] == read(run / 'COMPLETE.json')['training_updates'] == 0
    assert not list(run.rglob('optimizer.pt')) and not list(run.rglob('model.safetensors'))
    for value in config['models'].values():
        state = Path(value['model']) / 'training_state.json'
        assert sha(state) == value['training_state_sha256']
        assert read(state)['weight_digest'] == value['weight_digest']
    assert sha(config['initial_harness']) == config['initial_harness_sha256']
    datasets, task_ids = {}, set()
    for entry in config['data']['splits'].values():
        assert sha(entry['path']) == entry['sha256']
        rr = pd.read_parquet(entry['path']).to_dict('records')
        assert len(rr) == entry['count']
        ids = {r['extra_info']['task_id'] for r in rr}
        assert len(ids) == len(rr) and not ids & task_ids
        task_ids |= ids; datasets[entry['path']] = rr
    receipts = {p.stem: p for p in (run / 'pool').rglob('calls/*.json')}
    assert len(receipts) == sum(1 for _ in (run / 'pool').rglob('calls/*.json'))
    used, contexts, verified = set(), set(), {}
    for path in run.rglob('val.json'):
        if 'shards' in path.parts:
            continue
        value = read(path); meta = value['metadata']; rr = rows(path.parent / 'trajectories.jsonl')
        data = datasets[meta['dataset_path']]
        entry = next(e for e in config['data']['splits'].values() if e['path'] == meta['dataset_path'])
        assert meta['dataset_sha256'] == entry['sha256']
        assert meta['sampling_schedule'] == [[0, i] for i in range(len(data))]
        assert len(rr) == value['num_examples'] == len(data)
        assert meta['temperature'] == 1.0 and meta['assistant_token_budget'] == 2048
        for index, r in enumerate(rr):
            task = data[index]
            assert r['draw'] == 0 and r['example_index'] == index
            assert r['task_id'] == task['extra_info']['task_id']
            assert r['rollout_seed'] == meta['seed'] * 100003 + index
            assert r['sampling_identity']['weight_digest'] == meta['weight_digest']
            assert r['harness_sha256'] == meta['harness_sha256'] and r['answer_redacted']
            assert r['reward'] == binary_reward(r['last_assistant'], task['reward_model']['ground_truth']['target'])
            assert r['assistant_tokens'] == sum(len(c['response_ids']) for c in r['policy_calls']) <= 2048
            physical = Path(r['physical_worker_run'])
            for c in r['policy_calls']:
                key = c['call_id']; assert key not in used
                assert read(receipts[key]) == c == read(physical / 'calls' / (key + '.json'))
                assert c['sampling_identity']['weight_digest'] == meta['weight_digest']
                assert c['stop_at_chat_end'] and len(c['response_ids']) <= c['requested_max_tokens'] <= 1024
                assert len(c['response_ids']) == len(c['response_logprobs']) and np.isfinite(c['response_logprobs']).all()
                used.add(key)
            if r.get('context_rejection_id'):
                key = r['context_rejection_id']; assert key not in contexts
                recheck_context(physical / 'context_rejections' / (key + '.json')); contexts.add(key)
        assert sum(r['reward'] for r in rr) == value['solved_examples']
        assert value['success_rate'] == value['solved_examples'] / len(rr)
        verified[str(path.parent)] = value
    assert used == set(receipts)
    assert contexts == {p.stem for p in (run / 'pool').rglob('context_rejections/*.json')}
    budgets = []
    for number in [1, 2]:
        cycle = run / f'cycle_{number:02d}'; start = read(cycle / 'START.json')
        unique_new, old_unique = set(), set()
        for stage, old_stage in [('pre', 'editor_pre_proposals'), ('post', 'editor_post_proposals')]:
            directory = cycle / (stage + '_proposals')
            rr = rows(directory / 'scored.jsonl'); budget = read(directory / 'budget.json')
            receipt = read(directory / 'response_receipt.json')
            job = run / 'proposer_jobs' / receipt['job_id']
            assert sha(job / 'response.json') == receipt['response_sha256']
            assert len(rr) == config['proposal_slots'][stage] * 2
            assert budget['caps'] == config['new_evaluation_caps_per_group'][str(number)][stage]
            newly_seen = [0, 0]
            for r in rr:
                if r['valid_source']:
                    assert without_system(r['source']) == without_system(Path(start['harness']).read_text())
                if r['outcome_kind'] != 'valid':
                    assert r['reward_delta'] is None; continue
                candidate, base = r['reward_provenance']['candidate'], r['reward_provenance']['baseline']
                assert candidate in verified and base in verified
                v, b = verified[candidate], verified[base]
                assert v['metadata']['weight_digest'] == b['metadata']['weight_digest'] == start['solver']['weight_digest']
                assert v['metadata']['dataset_path'] == b['metadata']['dataset_path']
                assert v['metadata']['seed'] == b['metadata']['seed'] == 142 + number
                assert v['metadata']['harness_sha256'] == hashlib.sha256(r['source'].encode()).hexdigest()
                assert r['reward_delta'] == v['success_rate'] - b['success_rate']
                if candidate not in unique_new:
                    newly_seen[int(r['group'][-1])] += 1; unique_new.add(candidate)
            assert newly_seen == budget['new_evaluations']
            assert all(n <= cap for n, cap in zip(newly_seen, budget['caps']))
            old_unique |= {r['reward_provenance']['candidate'] for r in
                rows(reference / cycle.name / old_stage / 'scored.jsonl') if r['reward_provenance']['candidate']}
        assert len(unique_new) <= len(old_unique) == [11, 10][number - 1]
        decision = read(cycle / 'selection/decision.json')
        assert len(decision['shortlist_ids']) == 2 and not decision['test_feedback_used']
        eligible = {}; post = rows(cycle / 'post_proposals/scored.jsonl')
        for r in sorted((r for r in post if r['outcome_kind'] == 'valid'), key=lambda r: (-r['reward_delta'], r['id'])):
            eligible.setdefault(hashlib.sha256(r['source'].encode()).hexdigest(), r['id'])
        assert decision['shortlist_ids'] == list(eligible.values())[:2]
        for name, value in decision['results'].items():
            assert value == verified[str(cycle / 'selection/results' / name)]
            assert value['num_examples'] == 24 and value['metadata']['seed'] == 42
        best = sorted(decision['results'], key=lambda n: (-decision['results'][n]['success_rate'], decision['results'][n]['mean_turn_count']))[0]
        assert decision['accepted'] == best
        selected = cycle / 'selection/harnesses' / best / 'harness.py'
        assert sha(selected) == decision['selected_sha256'] == read(cycle / 'complete.json')['sha256']
        budgets.append({'cycle': number, 'new_candidate_dataset_evaluations': len(unique_new),
            'reference_candidate_dataset_evaluations': len(old_unique),
            'candidate_trajectories': len(unique_new) * 8, 'reference_candidate_trajectories': len(old_unique) * 8,
            'selection_trajectories': 72, 'baseline_trajectories': 16})
    stopped = read(run / 'pool/pool_stopped.json')
    assert len(stopped['workers']) == len(config['pool_gpus'])
    assert all(w['returncode'] == 0 for w in stopped['workers'])
    loads = []
    for directory in (run / 'pool/replicas').glob('slot_*'):
        assert read(directory / 'ready.json')['weight_digest'] == config['models']['W2_initial']['weight_digest']
        for file in directory.glob('checkpoint_loads/*.json'):
            proof = read(file)
            assert proof['all_parameter_hashes_equal'] and proof['weights'] in config['checkpoint_allowlist']
            assert proof['weight_digest'] == read(Path(proof['weights']) / 'training_state.json')['weight_digest']
            loads.append(str(file))
        service, stop = read(directory / 'services.json'), read(directory / 'services_stopped.json')
        assert all(service[k] == stop[k] for k in ['worker_pid', 'sandbox_pid'])
        assert all(stop[k] in (0, -15) for k in ['worker_returncode', 'sandbox_returncode'])
    value = {'passed_audit': True, 'new_trajectories': sum(v['num_examples'] for v in verified.values()),
        'unique_model_calls': len(used), 'context_rejections': len(contexts), 'training_updates': 0,
        'budget_comparison': budgets, 'single_draw_only': True, 'checkpoint_reload_proofs': loads,
        'services_stopped': True, 'source_and_checkpoint_states_unchanged': True,
        'test_not_used_for_selection': not read(run / 'SEARCH_COMPLETE.json')['test_feedback_used'],
        'note': 'Historical baseline reuse is checked against its previously audited file hash in report().'}
    write_json(run / 'verification.json', value)
    return value


def report(run, audit_result):
    config = read(run / 'config.json'); expanded = Path(config['expanded_reference'])
    curves = read(run / 'curve_sources.json'); stats = []; comparisons = []
    def load_cohorts(sources):
        result = []
        for cohort in config['cohorts']:
            source = sources[cohort]; path = Path(source['path'])
            assert sha(path / 'trajectories.jsonl') == source['trajectories_sha256']
            rr = rows(path / 'trajectories.jsonl')
            rr = [r for r in rr if r['draw'] == 0]
            assert len(rr) == config['data']['splits'][cohort]['count']
            result.extend(rr)
        assert len(result) == 173
        return result
    previous = None
    for point in curves:
        prior_checks = read(expanded / 'verification.json')['summaries'][point['model']]
        for cohort, check in prior_checks.items():
            assert sha(expanded / 'eval' / point['model'] / cohort / 'trajectories.jsonl') == check['trajectories_sha256']
        rr = load_cohorts(point['cohorts'])
        baseline = load_cohorts({c: {'path': str(expanded / 'eval' / point['model'] / c),
            'trajectories_sha256': sha(expanded / 'eval' / point['model'] / c / 'trajectories.jsonl')}
            for c in config['cohorts']})
        assert all(r['sampling_identity']['weight_digest'] == point['weight_digest'] and
            r['harness_sha256'] == point['harness_sha256'] for r in rr)
        assert [r['task_id'] for r in rr] == [r['task_id'] for r in baseline]
        assert all(r['rollout_seed'] == b['rollout_seed'] for r, b in zip(rr, baseline))
        solved, old = sum(r['reward'] for r in rr), sum(r['reward'] for r in baseline)
        wins = sum(r['reward'] > b['reward'] for r, b in zip(rr, baseline))
        losses = sum(r['reward'] < b['reward'] for r, b in zip(rr, baseline))
        discordant = wins + losses
        p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
        item = {'label': point['label'], 'model': point['model'], 'solved': int(solved), 'total': 173,
            'percent': solved / 173 * 100, 'internal_solved': int(old), 'internal_percent': old / 173 * 100,
            'gain_vs_internal_pp': (solved - old) / 173 * 100,
            'newly_solved': wins, 'regressed': losses, 'mcnemar_exact_two_sided_p': p,
            'mean_assistant_tokens': sum(r['assistant_tokens'] for r in rr) / 173,
            'harness_sha256': point['harness_sha256'],
            'phase_gain_pp': None if previous is None else (solved - previous) / 173 * 100}
        stats.append(item); previous = solved
    summary = {'stages': stats, 'audit': audit_result, 'elapsed_seconds': read(run / 'COMPLETE.json')['elapsed_seconds'],
        'search_seconds': read(run / 'SEARCH_COMPLETE.json')['elapsed_seconds'],
        'limitations': config['limitations'], 'budget_meaning': config['budget_meaning']}
    write_json(run / 'statistics.json', summary)
    labels = ['起点', '第1轮 harness 后', '换入 W3 后', '第2轮 harness 后', '换入 W4 后']
    lines = ['# 外部改写器：同预算 harness 搜索与冻结权重回放', '',
        '本轮只改 harness，所有求解器权重固定，训练更新为 0。当前 Codex 会话通过本机 FastAPI 队列实际编写候选。', '',
        '| 阶段 | 权重 | 内部改写器的 H3 | 外部改写结果 | 同权重差值 |',
        '|---|---|---:|---:|---:|']
    for label, item in zip(labels, stats):
        lines.append(f"| {label} | {item['model']} | {item['internal_solved']}/173 ({item['internal_percent']:.2f}%) | {item['solved']}/173 ({item['percent']:.2f}%) | {item['gain_vs_internal_pp']:+.2f} pp |")
    final = stats[-1]
    lines += ['', f"最终同 W4 比较：新增解对 {final['newly_solved']} 题，退步 {final['regressed']} 题，净变化 {final['solved'] - final['internal_solved']:+d} 题；精确配对 McNemar p={final['mcnemar_exact_two_sided_p']:.4f}。",
        '', '每题仅使用 seed 42 的第 0 次采样，temperature=1.0，总输出 2048 token、每次调用 1024 token。历史基线直接复用同一采样身份的已审计记录，没有将本次单次成绩与历史 mean@8 混比。',
        '', '## 预算与选择', '',
        '每轮 16 个初步提案、8 个反馈后提案。按原实验缓存去重后的实际次数，两轮分别最多允许 11、10 个不同候选—开发题包评测，每个题包 8 题。第一轮后阶段只能复用已评候选，第二轮允许两个新组合。最后各选 2 个候选与 incumbent 一起在独立的 24 题开发选择集上按原 WHALE 规则选择。',
        '', '所有修改仅限 SYSTEM_PROMPT，保留现有 H3 的 stop/recovery hooks 和 MAX_TURNS。外部改写器的思考 token 与费用无法与内部 2B 等价计量；这里匹配的是提案额度和求解器开发采样预算，不是总 FLOPs 或美元。', '',
        f"本轮新增 {audit_result['new_trajectories']} 条任务轨迹、{audit_result['unique_model_calls']} 次真实模型调用；{len(config['pool_gpus'])} 张空闲 GPU。执行总耗时 {summary['elapsed_seconds']/60:.1f} 分钟，含活跃会话提案等待，不含前置实现和最终审计。", '',
        '## 解释范围', '',
        'W3/W4 是旧 H3 下训练得到的 checkpoint，此曲线是固定权重的 harness 替换与权重回放，不能称为新 harness 下重新交替训练的结果。全部 173 题有历史使用记录，不是新的独立测试集。两轮选择完成后才运行本次测试，测试成绩不参与候选修改或选择。', '',
        '已检查原始服务回执、逐条奖励、采样 seed、预算、代码修改范围、checkpoint 重载及服务关闭。详见 verification.json、statistics.json 与 PROTOCOL.json。', '']
    selections = []
    for number in [1, 2]:
        cycle = f'cycle_{number:02d}'
        decision = read(run / cycle / 'selection/decision.json')
        old_decision = read(Path(config['reference']) / cycle / 'selection/decision.json')
        candidates = [v for k, v in decision['results'].items() if k != 'incumbent']
        best_candidate = sorted(candidates, key=lambda v: (-v['success_rate'], v['mean_turn_count']))[0]
        old_best = max(v['solved_examples'] for k, v in old_decision['results'].items() if k != 'incumbent')
        selections.append({'cycle': number, 'incumbent_solved': decision['results']['incumbent']['solved_examples'],
            'internal_best_candidate_solved': old_best, 'external_best_candidate_solved': best_candidate['solved_examples'],
            'incumbent_mean_calls': decision['results']['incumbent']['mean_turn_count'],
            'external_best_mean_calls': best_candidate['mean_turn_count'],
            'accepted': decision['accepted'], 'changed': decision['changed']})
    lines += ['## 开发选择详情', '', '| 轮次 | incumbent | 原内部候选最佳 | 本次外部候选最佳 | 接受新代码 |',
        '|---|---:|---:|---:|---|']
    for item in selections:
        lines.append(f"| {item['cycle']} | {item['incumbent_solved']}/24 | {item['internal_best_candidate_solved']}/24 | {item['external_best_candidate_solved']}/24 | {'是' if item['changed'] else '否'} |")
    lines += ['', '解题率相同时，原 WHALE 选择规则优先平均调用轮数更少的候选；接受候选不必然表示开发集多解了题。原内部候选只引用历史结果，未额外重新评测。', '']
    reproduction = read(run / 'BASELINE_REPRODUCTION.json')
    same = sum(item['same_rewards'] for item in reproduction)
    count = sum(item['total'] for item in reproduction)
    lines += [f'同一模型、代码和 seed 的开发基线复现：{same}/{count} 条奖励与历史相同。其余轨迹的初始 prompt token 相同，但生成 token 出现差异；因此单个 seed 不等于跨运行完全确定，细小差异需要谨慎解释。本轮不追加重复 seed。', '']
    summary['development_selections'] = selections
    summary['baseline_reproduction'] = reproduction
    write_json(run / 'statistics.json', summary)
    (run / 'REPORT.md').write_text('\n'.join(lines))
    write_json(run / 'STATUS.json', {'stage': 'complete_audited', 'final_solved': final['solved'],
        'internal_final_solved': final['internal_solved'], 'gain_pp': final['gain_vs_internal_pp']})
    print(json.dumps({'passed_audit': True, 'stages': stats, 'elapsed_minutes': summary['elapsed_seconds']/60}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve(); report(run, audit(run))
