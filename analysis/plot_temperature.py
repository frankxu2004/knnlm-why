import matplotlib.pyplot as plt

# names = ['real_mask', 'faiss_flat_mask', 'faiss_mask', 'faiss_mask_nprobe', 'faiss_mask_recomp']
names = ['faiss_flat_mask', 'faiss_mask', 'faiss_mask_recomp']

labels = ['real mask, real score', 'FAISS mask, FAISS score', 'FAISS mask, real score']
fig, ax = plt.subplots()

for name, l in zip(names, labels):
    x = []
    y = []
    with open(f'{name}_softmax_temperature.txt', 'r') as infile:
        for line in infile:
            temperature = float(line.split('\t')[1])
            if temperature == 0. or temperature > 3.01:
                continue
            best_ppl = float(line.split('\t')[4])
            x.append(temperature)
            y.append(best_ppl)
    ax.scatter(x, y, label=l, alpha=0.9, edgecolors='none')

ax.legend()
ax.grid(True)

plt.savefig('temperature.png', dpi=300)
