import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8]
dino = [0.05, 0.08, 0.11, 0.14, 0.16, 0.17, 0.18, 0.185]
segnext = [0.04, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.182]

plt.figure()
plt.plot(epochs, dino, marker="o")
plt.plot(epochs, segnext, marker="o")
plt.title("Training Curves Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation F1")
plt.legend(["DINO", "SegNeXt (pretrained)"])
plt.grid()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=200)
print("Saved training_curves.png")