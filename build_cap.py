"""Build the Danenberg graphs at a larger per-image cell cap, under the _c<n> suffix.

python build_cap.py <n_cells>
"""
import os, sys, shutil
n = int(sys.argv[1])
G = os.environ.get('TDA_DATA', 'data') + '/graphs'
tmp = os.path.join(G, 'tmp_c%d' % n)
os.makedirs(tmp, exist_ok=True)
os.environ['TDA_GRAPHS_OUT'] = tmp
sys.argv = ['build_graphs.py', 'danenberg', str(n)]
import build_graphs
build_graphs.main()
os.replace(os.path.join(tmp, 'danenberg_graphs.pkl'), os.path.join(G, 'danenberg_graphs_c%d.pkl' % n))
shutil.rmtree(tmp, ignore_errors=True)
print('BUILD_CAP_DONE', n, flush=True)
