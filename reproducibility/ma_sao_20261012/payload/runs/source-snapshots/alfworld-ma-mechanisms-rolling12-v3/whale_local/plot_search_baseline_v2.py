"""Plot audited raw alternation and fixed-axis comparisons, never best-so-far."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
run=args.run.resolve();audit=json.loads((run/'verification.json').read_text());assert audit['passed_audit']
labels=['W0H0','W1H0','W1H1','W2H1','W2H2'];scores=audit['scores']
values=[100*scores[k]['solved']/scores[k]['total'] for k in labels]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(left,right)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.4,1]})
colors=['#2563eb','#d97706','#2563eb','#d97706']
for i,color in enumerate(colors):
    left.plot([i,i+1],values[i:i+2],color=color,lw=2.5)
left.scatter(range(5),values,s=55,color='#172554',zorder=3)
for i,(name,value) in enumerate(zip(labels,values)):
    left.annotate(f"{scores[name]['solved']}/{scores[name]['total']}\n{value:.1f}%",(i,value),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
left.set_xticks(range(5),labels);left.set_ylabel('Success rate (%)');left.set_ylim(0,100)
left.grid(axis='y',alpha=.2);left.set_title('Raw alternation trajectory',loc='left',weight='bold')
left.legend(handles=[Line2D([0],[0],color='#2563eb',lw=2.5,label='Weight update'),Line2D([0],[0],color='#d97706',lw=2.5,label='Harness update')],loc='lower right')
keys=['W0H2','W2H0','W2H2'];v=[100*scores[k]['solved']/scores[k]['total'] for k in keys]
right.bar(range(3),v,color=['#93c5fd','#fdba74','#334155'],width=.62)
for i,(name,val) in enumerate(zip(keys,v)):right.text(i,val+2,f"{scores[name]['solved']}/{scores[name]['total']}\n{val:.1f}%",ha='center',fontsize=10)
right.set_xticks(range(3),['Base W\nFinal H','Final W\nBase H','Final W\nFinal H']);right.set_ylim(0,100)
right.set_title('Contributions at the final state',loc='left',weight='bold');right.grid(axis='y',alpha=.2);right.set_axisbelow(True)
fig.suptitle('WHALE / HotpotQA distractor — corrected training',x=.07,ha='left',fontsize=16,weight='bold')
fig.text(.07,.025,'Fresh development set: 64 questions, one draw, seed 42. Test remains unopened. Local BM25 adaptation.',fontsize=10,color='#475569')
fig.tight_layout(rect=[0,.08,1,.92])
for ext in ['png','pdf','svg']:fig.savefig(run/f'alternation_curve.{ext}',dpi=180,bbox_inches='tight')
# Orthogonal curves reuse existing single-draw cells; no extra model sampling.
fig2,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
for ax,keys,title,color in [
    (axes[0],['W0H0','W1H0','W2H0'],'Weight changes / fixed initial harness','#2563eb'),
    (axes[1],['W2H0','W2H1','W2H2'],'Harness changes / fixed final weights','#d97706')]:
    y=[100*scores[k]['solved']/scores[k]['total'] for k in keys]
    ax.plot(range(3),y,'o-',color=color,lw=2.5)
    for i,(k,value) in enumerate(zip(keys,y)):
        ax.annotate(f"{scores[k]['solved']}/{scores[k]['total']}",(i,value),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set_xticks(range(3),keys);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_title(title,fontsize=11)
axes[0].set_ylabel('Success rate (%)')
fig2.suptitle('Fixed-axis attribution — no additional evaluation draws',fontsize=14,weight='bold')
fig2.text(.08,.025,'64 development questions, seed 42. All plotted points are raw observed scores. Not an independent test result.',fontsize=9,color='#475569')
fig2.tight_layout(rect=[0,.07,1,.94])
for ext in ['png','pdf','svg']:fig2.savefig(run/f'fixed_axis_curves.{ext}',dpi=180,bbox_inches='tight')
