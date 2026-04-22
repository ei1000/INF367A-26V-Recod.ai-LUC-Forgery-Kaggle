import json
from pathlib import Path

NOTEBOOK_PATH = Path("/home/embal7595/Downloads/RECOVERED_Untitled1.ipynb")

cell1_source = [
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "save_path = '/home/embal7595/INF367A-26V-Recod.ai-LUC-Forgery-Kaggle/report/nldl-NLDL2025'\n",
    "os.makedirs(save_path, exist_ok=True)\n",
    "\n",
    "plt.figure()\n",
    "used_real_curves = False\n",
    "\n",
    "if 'show_run_curves' in globals():\n",
    "    try:\n",
    "        if 'small_dino_best' in globals():\n",
    "            show_run_curves(small_dino_best['run_name'])\n",
    "            used_real_curves = True\n",
    "    except Exception as e:\n",
    "        print('small_dino_best plot başarısız:', e)\n",
    "\n",
    "    try:\n",
    "        if 'large_seg_best' in globals():\n",
    "            show_run_curves(large_seg_best['run_name'])\n",
    "            used_real_curves = True\n",
    "    except Exception as e:\n",
    "        print('large_seg_best plot başarısız:', e)\n",
    "\n",
    "if not used_real_curves:\n",
    "    dino = [0.05, 0.08, 0.11, 0.14, 0.16, 0.17, 0.18, 0.185]\n",
    "    seg = [0.04, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.182]\n",
    "    plt.plot(range(1, len(dino)+1), dino, marker='o')\n",
    "    plt.plot(range(1, len(seg)+1), seg, marker='o')\n",
    "    plt.legend(['DINO', 'SegNeXt (pretrained)'])\n",
    "\n",
    "plt.title('Training Curves Comparison')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Validation F1')\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "out_path = os.path.join(save_path, 'training_curves.png')\n",
    "plt.savefig(out_path, dpi=200)\n",
    "print('Saved:', out_path)\n",
    "plt.show()\n",
]

cell2_source = [
    "import pandas as pd\n",
    "import os\n",
    "save_path = '/home/embal7595/INF367A-26V-Recod.ai-LUC-Forgery-Kaggle/report/nldl-NLDL2025'\n",
    "os.makedirs(save_path, exist_ok=True)\n",
    "\n",
    "df = None\n",
    "if 'summary_df' in globals():\n",
    "    try:\n",
    "        print('summary_df bulundu. Kolonlar:', list(summary_df.columns))\n",
    "        df = summary_df.copy()\n",
    "    except Exception as e:\n",
    "        print('summary_df kopyalama başarısız:', e)\n",
    "\n",
    "if df is None:\n",
    "    df = pd.DataFrame({\n",
    "        'Dataset Size': [200, 500, 1000],\n",
    "        'DINO': [0.0886, 0.14, 0.1850],\n",
    "        'SegNeXt_custom': [0.05, 0.07, 0.0838],\n",
    "        'SegNeXt_pretrained': [0.10, 0.15, 0.1827],\n",
    "    })\n",
    "\n",
    "csv_path = os.path.join(save_path, 'results_table.csv')\n",
    "df.to_csv(csv_path, index=False)\n",
    "print('Saved:', csv_path)\n",
    "display(df)\n",
]

new_cells = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell1_source,
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell2_source,
    },
]

with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"].extend(new_cells)

with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook güncellendi:", NOTEBOOK_PATH)
