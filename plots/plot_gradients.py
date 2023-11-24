import matplotlib.pyplot as plt
import numpy as np

p_margin_mean, p_margin_std = np.asarray([0.1887, 0.0394]), [0.1425, 0.1346]
l_margin_mean, l_margin_std = np.asarray([0.8125, 0.1875]), [0.3966, 0.3966]
l_margin_2 = [], []

logits = np.asarray([3.6362, -3.3722,  0.2788, -0.5738])
probs = np.asarray([0.8665, 0.0451, 0.0569, 0.0315])


fig, ax = plt.subplots(1)

x = np.asarray([0, 1])
width = 0.25  # the width of the bars

results = np.stack((p_margin_mean, l_margin_mean), 0)

print(results)
offset = width * 0
rects = ax.bar(x, results[0, :], width, label='Margin on probabilities (m=0.3)')
# ax.bar_label(rects, padding=3)

rects = ax.bar(x + width, results[1, :], width,  label='Margin on logits (m=1)')
# ax.bar_label(rects, padding=3)

ax.set_xticks(x + (width / 2), ['Class 0', 'Class 1'])
ax.legend(loc='upper right', ncols=1)
ax.set_ylabel('Gradients magnitude')

fig.savefig('gradients.pdf',  bbox_inches='tight')
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1)

x = np.asarray([0, 1, 3, 4], dtype='float')
width = 0.1  # the width of the bars

results = np.stack((p_margin_mean, l_margin_mean), 0)
color = ['red', 'blue', 'purple', 'red']
legend = ['Task 0', 'Task 1']

print(results)
offset = width * 0
rects = ax1.bar(x[:2], logits[:2], width=1.0, alpha=0.8, color='red')
rects = ax1.bar(x[2:], logits[2:], width=1.0, alpha=0.8, color='blue')
ax1.hlines(0, -0.48, 4.48, colors='k', linewidths=0.8)
# ax.bar_label(rects, padding=3)
ax1.set_xticks([], [])

# rects = ax2.bar(x, probs, width=1.0, color=color)
rects = ax2.bar(x[:2], probs[:2], width=1.0, alpha=0.8, color='red')
rects = ax2.bar(x[2:], probs[2:], width=1.0, alpha=0.8,  color='blue')
# ax.bar_label(rects, padding=3)

ax2.set_xticks(x + (width / 2), ['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# plt.legend(loc='upper right', ncols=1)
# ax.set_ylabel('Gradients magnitude')

handles = [plt.Rectangle((0,0),1,1, color=c) for c in ['red', 'blue']]
plt.legend(handles, ['Classes from task 0', 'Classes from task 1'])
fig.savefig('logits.pdf',  bbox_inches='tight')

plt.show()

