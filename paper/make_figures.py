"""Vector figures for the manuscript (harness; matplotlib allowed)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42})
INK = '#1a1a1a'; INK2 = '#55534e'

# Fig 1: pipeline
fig, a = plt.subplots(figsize=(6.8, 1.55))
a.set_xlim(0, 100); a.set_ylim(4, 24); a.axis('off')
steps = [('Rectified\nstereo pair', '', '#f1f0ec'), ('Siamese\nfeatures', '1/4 res.,\nshared', '#cde2fb'),
         ('Cost volume', '|f$_L$-f$_R$|,\n12 levels', '#b7d3f6'), ('3D filter +\nsoft-argmin', 'sub-pixel\ndisparity', '#9ec5f4'),
         ('Guided\nrefine x2', 'dilated\nconvs', '#86b6ef'), ('Full-res\nrefinement', 'band SGM,\nLR, planes', '#d9d7cf'),
         ('Mesh +\nGaussians', 'Delaunay,\nclosed-form $\\Sigma$', '#fde3d6')]
n = len(steps); w = 11.8; gap = (100 - n * w) / (n - 1)
for i, (t, sub, col) in enumerate(steps):
    x = i * (w + gap)
    a.add_patch(FancyBboxPatch((x, 7), w, 15, boxstyle='round,pad=0,rounding_size=1.2', fc=col, ec='#bdbab2', lw=0.7))
    a.text(x + w / 2, 18.0, t, ha='center', va='center', fontsize=7, color=INK, fontweight='bold')
    if sub: a.text(x + w / 2, 11.0, sub, ha='center', va='center', fontsize=6.2, color=INK2)
    if i < n - 1:
        a.annotate('', xy=(x + w + gap - 0.3, 14.5), xytext=(x + w + 0.3, 14.5), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.8))
fig.savefig('figures/fig1_pipeline.pdf', bbox_inches='tight', pad_inches=0.02)

# Fig 2: test EPE progression (single series, direct labels)
labels = ['Synthetic\nonly', 'Real\nfine-tune', 'Search\n8$\\to$12', 'Lower\nLR', '+13 real\nscenes', '+ full-res\nrefinement']
vals = [5.21, 3.49, 2.47, 2.45, 2.29, 1.14]
fig, ax = plt.subplots(figsize=(5.2, 2.2))
cols = ['#3a7fd0'] * 5 + ['#0d366b']
b = ax.bar(range(6), vals, color=cols, width=0.62)
for i, v in enumerate(vals): ax.text(i, v + 0.08, '%.2f' % v, ha='center', va='bottom', fontsize=7.5, color=INK)
ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('Test EPE (px), lower is better', fontsize=7.5)
ax.set_ylim(0, 6); ax.grid(axis='y', color='#e6e4df', lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.spines['left'].set_color('#9a978f'); ax.spines['bottom'].set_color('#9a978f'); ax.tick_params(colors=INK2)
fig.savefig('figures/fig2_progression.pdf', bbox_inches='tight', pad_inches=0.02)
print('ok')
