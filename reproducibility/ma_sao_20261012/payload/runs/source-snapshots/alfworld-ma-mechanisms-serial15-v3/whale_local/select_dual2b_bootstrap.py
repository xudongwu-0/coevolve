"""Select an editor using code behavior development checks, before task test."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
from pipeline import ROOT, sha, write_json

def read(p):return json.loads(Path(p).read_text())

def score(path):
    proof=read(path);assert proof['count']==20
    checks=[]
    for r in proof['results']:
        family=r['id'].split('-')[0]
        if family not in {'finalize','repair'}:continue
        require_recovery=family=='finalize' or '-preserve-' in r['id']
        require_repair=family=='repair' or '-preserve-' in r['id']
        passed=r['passed'] and (not require_recovery or r['recovers_partial_response']) and (not require_repair or r['tool_error_nudge'])
        checks.append({'id':r['id'],'passed_requested_behavior':bool(passed)})
    assert len(checks)==8
    return {'code_contract_passed':proof['passed'],'code_checks':20,
        'requested_hook_behavior_passed':sum(x['passed_requested_behavior'] for x in checks),
        'hook_checks':8,'details':checks,'checks_sha256':sha(path)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    candidates=['editor_continuation','editor_from_base']
    scores={name:score(run/(name+'_probe')/'executable_probes.json') for name in candidates}
    scores['base']=score(run/'editor_base_probe/executable_probes.json')
    chosen=max(candidates,key=lambda name:(scores[name]['requested_hook_behavior_passed'],scores[name]['code_contract_passed'],-candidates.index(name)))
    choice={'scores':scores,'chosen':chosen,'eligible':scores[chosen]['requested_hook_behavior_passed']==8,
        'development_only':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(run/'BOOTSTRAP_SELECTION.json',choice);print(json.dumps(choice))
    if not choice['eligible']:return
    old=ROOT/'runs/whale_dual2b_20260916';assert not (old/'test').exists() and not (run/'test').exists()
    c=read(old/'config.json');directory=run/'data';directory.mkdir()
    src=Path(c['data']['splits']['test']['path']);dst=directory/'fresh_test.parquet';shutil.copy2(src,dst)
    assert sha(src)==sha(dst)
    c['data']['splits']['test']['path']=str(dst)
    c.update(editor_initial_checkpoint=str(run/chosen/'checkpoint'),editor_focus_mode='telemetry_conditioned',
        require_harness_improvement_before_test=True,proposal_seed_policy='271 + sha256(phase_name)[:8] modulo 100000 + request index',
        bootstrap_selection=str(run/'BOOTSTRAP_SELECTION.json'))
    c.pop('bootstrap_total_epochs',None)
    write_json(run/'config.json',c);write_json(run/'eval_hardware.json',read(old/'eval_hardware.json'))
    write_json(run/'PROTOCOL.json',{'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reference_verification_sha256':read(old/'PROTOCOL.json')['reference_verification_sha256'],
        'test_count':32,'test_draws':4,'bootstrap_protocol_sha256':sha(run/'BOOTSTRAP_PROTOCOL.json'),
        'bootstrap_selection_sha256':sha(run/'BOOTSTRAP_SELECTION.json'),'aborted_prior_run':str(old),
        'aborted_prior_run_test_unopened':True,'bounded_telemetry_focus_not_open_ended_diagnosis':True})
    write_json(run/'PROTOCOL_AMENDMENT_BEFORE_MH.json',{'config_sha256':sha(run/'config.json'),
        'test_still_unopened':True,'reason':'Selected focused editor by predeclared development code behavior checks'})
    links=run/'checkpoints';links.mkdir()
    reference=Path(c['solver_prefix_reference'])
    for name,target in [('harness_E0',Path(c['editor_initial_checkpoint'])),('solver_W0',Path(c['base_model'])),
        ('solver_W1',reference/'round_01/checkpoint')]: (links/name).symlink_to(target,target_is_directory=True)
    write_json(run/'CHECKPOINT_ROLES.json',{'harness':str(links/'harness_E0'),'solver':str(links/'solver_W1'),
        'optimizer_states_separate':True,'base':str(links/'solver_W0')})
    snap=run/'source_snapshots/online';snap.mkdir(parents=True)
    manifest={}
    sources=[p for p in (ROOT/'whale_local').glob('*.py') if not p.name.startswith(('audit_','verify_','assess_'))]
    sources += [ROOT/p for p in ['WHALE/domains/math_reasoning/verl/workers/actor/dp_actor.py',
        'WHALE/domains/math_reasoning/environments/retool/base_harness.py',
        'WHALE/domains/math_reasoning/meta_harness/meta_harness_retool.py',
        'WHALE/domains/math_reasoning/meta_harness/benchmark.py']]
    for source in sources:
        shutil.copy2(source,snap/source.name);manifest[str(source)]=sha(source)
    write_json(run/'ONLINE_SOURCE_MANIFEST.json',manifest)

if __name__=='__main__':main()
