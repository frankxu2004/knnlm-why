import matplotlib.pyplot as plt

names = ['real_mask', 'faiss_flat_mask', 'faiss_mask', 'faiss_mask_nprobe', 'faiss_mask_recomp']
fig, ax = plt.subplots()

for name in names:
    x = []
    y = []
    with open(f'{name}_softmax_temperature.txt', 'r') as infile:
        for line in infile:
            temperature = float(line.split('\t')[1])
            if temperature == 0. or temperature >= 6:
                continue
            best_ppl = float(line.split('\t')[4])
            x.append(temperature)
            y.append(best_ppl)
    ax.scatter(x, y, label=name, alpha=0.9, edgecolors='none')

ax.legend()
ax.grid(True)

plt.savefig('temperature.png', dpi=300)
