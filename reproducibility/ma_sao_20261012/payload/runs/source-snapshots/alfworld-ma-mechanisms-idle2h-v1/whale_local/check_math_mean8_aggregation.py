"""CPU check that repeated draws are averaged and paired by question."""
import copy
import json

from pipeline import ROOT, sha, write_json
from verify_math_gate import repeated_outcomes


def main():
    ids = ['question-b', 'question-a']
    rows = [{'task_id': ids[index], 'example_index': index, 'draw': draw,
             'reward': int(draw < (1 if index == 0 else 7))}
            for draw in range(8) for index in range(2)]
    scores, means = repeated_outcomes(rows, ids, 8)
    assert means == {'question-b': .125, 'question-a': .875}
    assert scores['mean_accuracy_percent'] == 50
    assert scores['successful_draws'] == 8 and scores['total_draws'] == 16
    # At least one success would incorrectly score 100%; the last draw would
    # incorrectly score 0%. Both questions end in failure in this fixture.
    assert all(any(r['reward'] for r in rows if r['task_id'] == key) for key in ids)
    assert [r['reward'] for r in rows[-2:]] == [0, 0]
    malformed = [rows[:-1]]
    duplicate = copy.deepcopy(rows)
    duplicate[-1] = copy.deepcopy(duplicate[-2])
    malformed.append(duplicate)
    nonbinary = copy.deepcopy(rows)
    nonbinary[0]['reward'] = .5
    malformed.append(nonbinary)
    for bad in malformed:
        try:
            repeated_outcomes(bad, ids, 8)
        except AssertionError:
            pass
        else:
            raise AssertionError('Malformed repeated evaluation was accepted')
    result = {'passed': True, 'scope': 'Synthetic CPU metric check; no benchmark evaluation',
        'question_means': means, 'mean_accuracy_percent': scores['mean_accuracy_percent'],
        'distinguishes_mean_at_8_from_pass_at_8_and_last_draw': True,
        'missing_duplicate_and_nonbinary_draws_rejected': True,
        'source_sha256': {name: sha(ROOT / 'whale_local' / name) for name in
                          ['check_math_mean8_aggregation.py', 'verify_math_gate.py']}}
    write_json(ROOT / 'audit/whale/MATH_MEAN8_AGGREGATION_CHECK_20260915.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
