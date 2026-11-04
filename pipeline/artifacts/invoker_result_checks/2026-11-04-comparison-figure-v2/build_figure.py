"""Create an exclusive static figure from saved method-level comparison inputs."""
import argparse, csv, hashlib, json, math, platform, shutil, subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def key(repo, file, line):
    return repo, file.replace('\\', '/').removeprefix(repo.replace('/', '_') + '/'), int(line)

def lens_area(distance, r1, r2):
    if distance >= r1 + r2: return 0.0
    if distance <= abs(r1-r2): return math.pi * min(r1,r2)**2
    a = math.acos((distance**2 + r1**2-r2**2)/(2*distance*r1))
    b = math.acos((distance**2 + r2**2-r1**2)/(2*distance*r2))
    c = math.sqrt((-distance+r1+r2)*(distance+r1-r2)*(distance-r1+r2)*(distance+r1+r2))
    return r1*r1*a + r2*r2*b - c/2

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
base=args.repo_root.resolve()/'pipeline/artifacts'
inputs=[base/'invoker_result_checks/2026-11-04-independent-v2/invokers.csv', base/'pym2t_focal_tests.csv']
before={str(p.relative_to(args.repo_root.resolve())):sha(p) for p in inputs}
with inputs[0].open(encoding='utf-8-sig',newline='') as stream:
    inv=list(csv.DictReader(stream))
pop={key(r['repo'],r['file'],r['line']) for r in inv}
ind={key(r['repo'],r['file'],r['line']) for r in inv if r['has_result_check_candidate']=='1'}
pym=set()
with inputs[1].open(encoding='utf-8-sig',newline='') as stream:
    for r in csv.DictReader(stream):
        if r['focal_method'] and r['focal_line']:
            method=key(r['repo'],r['focal_file'],r['focal_line'])
            if method in pop: pym.add(method)
n=len(pop); a=len(pym); b=len(ind); both=len(pym & ind)
assert n and a and b and 0 < both < min(a,b)
left=len(pym-ind); right=len(ind-pym); outside=len(pop-(pym|ind)); union=len(pym|ind)
assert left+both+right+outside==n
out=args.output_dir.resolve()
out.mkdir(parents=True,exist_ok=False)
shutil.copyfile(Path(__file__),out/'build_figure.py')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'svg.fonttype':'none','font.size':10,'axes.unicode_minus':False})
blue='#2877B8'; orange='#D9822B'; ink='#172B40'; muted='#526274'; grid='#DFE5EB'
fig=plt.figure(figsize=(12.8,5.6),facecolor='white')
fig.text(.045,.94,'Focal mappings and result-check evidence',fontsize=21,fontweight='bold',color=ink)
fig.text(.045,.885,f'Common population: {n:,} direct invoker methods',fontsize=11,color=muted)
fig.text(.045,.765,'A  Share of all direct invokers',fontsize=12,fontweight='bold',color=ink)
fig.text(.57,.765,'B  Overlap of the detected method sets',fontsize=12,fontweight='bold',color=ink)
ax=fig.add_axes([.235,.205,.275,.465])
values=[100*a/n,100*b/n]
ax.barh([1,0],values,height=.34,color=[blue,orange],zorder=3)
ax.set_yticks([1,0],['PyMethods2Test\nfocal-method mapping','Independent\nreturn-to-check evidence'],fontsize=10.5,color=ink)
ax.tick_params(axis='y',length=0,pad=12)
limit=max(values)*1.65
ax.set_xlim(0,limit); ax.set_ylim(-.6,1.6)
ax.set_xticks([0,1,2,3],[f'{v:.0f}%' for v in [0,1,2,3]])
ax.tick_params(axis='x',labelsize=9.5,colors=muted,length=0,pad=8)
ax.set_xlabel('Share of all direct invokers',fontsize=10,color=muted,labelpad=13)
ax.xaxis.grid(True,color=grid,linewidth=.7,zorder=0)
for side in ('top','right','left','bottom'): ax.spines[side].set_visible(False)
for y,count,value in zip([1,0],[a,b],values):
    ax.text(value+.09,y,f'{count:,}  |  {value:.2f}%',va='center',ha='left',fontsize=11,fontweight='bold',color=ink)
venn=fig.add_axes([.555,.195,.415,.50])
r1=1.; r2=math.sqrt(b/a); target=math.pi*r1*r1*both/a
lo=abs(r2-r1); hi=r1+r2
for _ in range(90):
    mid=(lo+hi)/2
    if lens_area(mid,r1,r2)>target: lo=mid
    else: hi=mid
d=(lo+hi)/2
assert abs(lens_area(d,r1,r2)/math.pi*a-both)<1e-6
venn.add_patch(Circle((0,0),r1,facecolor=blue,edgecolor=blue,alpha=.35,linewidth=1.1))
venn.add_patch(Circle((d,0),r2,facecolor=orange,edgecolor=orange,alpha=.38,linewidth=1.1))
venn.set_aspect('equal'); venn.set_xlim(-r1-.16,d+r2+.16);venn.set_ylim(-r2-.15,r2+.15);venn.axis('off')
centers=[(-r1+(d-r2))/2,(d-r2+r1)/2,(r1+d+r2)/2]
for x,count,label in zip(centers,[left,both,right],['PyM only','Both','Independent only']):
    venn.text(x,.10,f'{count:,}',ha='center',va='center',fontsize=22,fontweight='bold',color=ink)
    venn.text(x,-.16,label,ha='center',va='center',fontsize=9.3,color=ink)
fig.text(.762,.145,f'Detected by either: {union:,} methods',ha='center',fontsize=10.5,color=ink)
fig.text(.762,.102,f'Outside both sets: {outside:,} ({100*outside/n:.2f}%)',ha='center',fontsize=10,color=muted)
fig.savefig(out/'invoker-method-comparison.png',dpi=300,facecolor='white')
fig.savefig(out/'invoker-method-comparison.svg',facecolor='white')
plt.close(fig)
with (out/'rates.csv').open('x',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['measure','methods','denominator','percent'])
    writer.writerows([['pymethods2test_focal_method',a,n,100*a/n],['independent_return_to_check_candidate',b,n,100*b/n]])
with (out/'overlap.csv').open('x',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['membership','methods','denominator','percent'])
    writer.writerows([[name,count,n,100*count/n] for name,count in [('pym_only',left),('both',both),('independent_only',right),('outside_both',outside)]])
caption=(f'Among {n:,} canonical observed direct invoker methods, PyMethods2Test identifies {a:,} focal-method correspondences ({100*a/n:.2f}%), while the independent saved-edge analysis detects returned-value-to-test-check candidates for {b:,} ({100*b/n:.2f}%). The sets overlap in {both:,} methods; {left:,} occur only in the focal mappings and {right:,} only in the independent set. {outside:,} methods are outside both detected sets. Both approaches are static and subject to false or missing associations. The independent analysis uses an incomplete saved first-edge call forest; correspondence is different from result checking, and neither establishes real-model execution, execution coverage, or semantic correctness. Test/support definitions remain in this historical all-direct-invoker population. Circle areas are proportional to set sizes and their intersection.\n')
with (out/'CAPTION.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write(caption)
after={str(p.relative_to(args.repo_root.resolve())):sha(p) for p in inputs}
assert before==after,'Saved input changed during rendering'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo_root,text=True).strip()
manifest={'revision':revision,'python':platform.python_version(),'matplotlib':matplotlib.__version__,
 'inputs':{name:{'sha256_before':before[name],'sha256_after':after[name]} for name in before},
 'counts':{'denominator':n,'pym_focal':a,'independent_candidate':b,'both':both,'pym_only':left,'independent_only':right,'outside_both':outside},
 'outputs':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
with (out/'manifest.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2,sort_keys=True);stream.write('\n')
print(str(out/'invoker-method-comparison.png'))
print(json.dumps(manifest['counts']))