import re

train_losses = [0] * 233
val_losses = [0] * 233

with open('checkpoints/wikitext103-bpe-overfit/overfit1.log') as logfile:
    for line in logfile:
        if line.startswith('epoch') and 'valid on' in line:
            epoch = int(re.search(r'epoch (\d+)', line).group(1))
            loss = float(re.search(r'loss ([-+]?\d*\.?\d+|\d+)', line).group(1))
            val_losses[epoch-1] = loss
        elif line.startswith('epoch'):
            epoch = int(re.search(r'epoch (\d+)', line).group(1))
            loss = float(re.search(r'loss ([-+]?\d*\.?\d+|\d+)', line).group(1))
            train_losses[epoch-1] = loss


print(train_losses)
print(val_losses)

with open('checkpoints/wikitext103-bpe-overfit/curve.csv', 'w') as outfile:
    for i in range(233):
        outfile.write('{}\t{}\t{}\n'.format(i+1, train_losses[i], val_losses[i]))
