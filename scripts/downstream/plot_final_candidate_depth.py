"""Plot verified original/adapted GDP depth profiles as standalone artifacts.

Activate pllm/protenix_abtcr; run:
python -B scripts/downstream/plot_final_candidate_depth.py --run ABS_VERIFIED_DEPTH_DIR
Writes PNG/SVG/PDF and source hashes, without changing any numerical results.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args();run = args.run.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert json.loads((run/'verification.json').read_text())['status'] == 'passed'
    assert json.loads((run/'status.json').read_text())['status'] == 'success'
    parent = ROOT/'output/diagnostics/fixed50000_depth_20261018/metrics.json'
    before, after = json.loads(parent.read_text()), json.loads((run/'metrics.json').read_text())
    names = [f'native_block{i}' for i in range(1,9)]+['native_final_norm']
    props = ['AC-SINS_pH7.4','HIC','PR_CHO','Titer','Tm2']
    fig, axs = plt.subplots(2,3,figsize=(13.2,7.4))
    for ax, prop in zip(axs.flat, props):
        values = lambda data: [data[n]['properties'][prop]['legacy_reference']['cv_mean_spearman'] for n in names]
        original, candidate = values(before),values(after)
        ax.plot(range(1,10),original,'o-',color='#6b7280',label='Original checkpoint',markersize=4)
        ax.plot(range(1,10),candidate,'o-',color='#147d92',label='Complementary teacher student',markersize=4)
        ax.axhline(original[1],color='#cf8723',linestyle='--',linewidth=1.2,label='Original block 2')
        ax.set_title(prop.replace('_',' '),loc='left',weight='bold')
        ax.set_xticks(range(1,10),[str(i) for i in range(1,9)]+['8+Norm'])
        ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel('Native block output');ax.set_ylabel('Mean fold Spearman')
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axs.flat[5].axis('off')
    handles, labels = axs.flat[0].get_legend_handles_labels()
    axs.flat[5].legend(handles,labels,loc='upper left',frameon=False,fontsize=10)
    axs.flat[5].text(.02,.60,'Verified controls\n\nBlocks 1–2: bitwise unchanged\nFinal output: saved features reproduced\n45 property/layer OOF checks\n\nSingle seed, exploratory reference CV.\nTm2 correlations remain weak.\nCross-task retention is a separate test.',
                     transform=axs.flat[5].transAxes,va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('GDPa1 depth profiles: original vs adapted model',fontsize=16,weight='bold',x=.05,ha='left')
    fig.tight_layout(rect=(0,0,1,.95))
    outputs = {}
    for extension in ['png','svg','pdf']:
        path = run/f'depth_profiles.{extension}'
        fig.savefig(path,dpi=180,bbox_inches='tight');outputs[str(path)] = sha(path)
    plt.close(fig)
    write(run/'plot_provenance.json',dict(source_sha256={str(p):sha(p) for p in [Path(__file__),parent,run/'metrics.json',run/'verification.json']},outputs=outputs))


if __name__ == '__main__':
    main()
