"""Run the commands of a worklist N at a time, with per-run logs and exit codes.

Each line of the list is tab-separated:  ENV<TAB>script.py<TAB>arg1<TAB>arg2 ...
where ENV is a comma-separated list of KEY=VALUE pairs (or '-' for none). Arguments are
passed as an argv list, never through a shell, so target names with ^, {, } or & are safe.
A run whose log already ends in a DONE marker is skipped, so a list can be resubmitted
after a wall-time kill and only the unfinished runs execute.

python runlist.py <list.tsv> [nproc]
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

LIST = sys.argv[1]
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 30
CODE = '.'
LOGS = 'logs/tp'
os.makedirs(LOGS, exist_ok=True)
DONE = ('_DONE', 'ROBUST_DONE', 'SCALING_DONE', 'EXPORT_DONE')


def parse(line):
    parts = line.rstrip('\n').split('\t')
    env = {}
    if parts[0] != '-':
        for kv in parts[0].split(','):
            k, v = kv.split('=', 1)
            env[k] = v
    return env, parts[1:]


def tag_of(env, argv):
    s = '_'.join(argv).replace(' ', '_').replace('/', '-')
    e = '_'.join('%s-%s' % (k, v.replace('/', '-')) for k, v in sorted(env.items()))
    return (s + ('__' + e if e else ''))[:180]


def finished(path):
    try:
        tail = open(path).read()[-400:]
    except OSError:
        return False
    return any(m in tail for m in DONE)


def run(job):
    env, argv = job
    tag = tag_of(env, argv)
    log = os.path.join(LOGS, tag + '.log')
    if finished(log) and not os.environ.get('TDA_NOSKIP'):
        print('%s skipped (done)' % tag, flush=True)
        return 0
    full = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', **env)
    t0 = time.time()
    with open(log, 'w') as f:
        p = subprocess.run([sys.executable, '-u'] + argv, cwd=CODE, env=full,
                           stdout=f, stderr=subprocess.STDOUT)
    print('%s rc=%d %.0fs' % (tag, p.returncode, time.time() - t0), flush=True)
    return p.returncode


jobs = [parse(l) for l in open(LIST) if l.strip() and not l.startswith('#')]
print('list %s has %d runs, %d at a time' % (LIST, len(jobs), NPROC), flush=True)
with ThreadPoolExecutor(max_workers=NPROC) as ex:
    rcs = list(ex.map(run, jobs))
print('LIST_DONE %s ok=%d fail=%d' % (LIST, sum(r == 0 for r in rcs),
                                      sum(r != 0 for r in rcs)), flush=True)
