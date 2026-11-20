# Sujet 5 — Vision Foundation Models and Self-Supervised Learning

Exposé du cours *LLM, LVM & GenAI — Fondations et Applications* (ENSAM Meknès, Université Moulay Ismail).
Groupe : NANKOULI Marc Thierry · DIALLO Souleymane · HINIMDOU Morsia Guitdam · CHAIBOU SAIDOU Abdoulaye — Encadrant : **Pr. ELANSARI**

## Livrables

| Livrable exigé | Fichier |
|---|---|
| Présentation (Beamer) | `presentation/presentation.pdf` (source `presentation.tex`) |
| Rapport scientifique PDF | `rapport/rapport.pdf` (source `rapport.tex`) |
| Code source | `notebook/demo_dinov2.ipynb` (toutes les figures et `results.json` en sont issus ; à ré-exécuter pour enregistrer les sorties dans le notebook, voir plus bas) |
| Démonstration pratique | section 10 du notebook (`search()`), figures dans `figures/` |
| Références | `references.bib` (articles originaux + documentations officielles) |

## Arborescence

```
Sujet5_VFM_SSL/
├── notebook/        demo_dinov2.ipynb, requirements.txt
├── figures/         figures PNG + results.json + results.tex (générés par le notebook)
├── tikz/            schémas d'architecture TikZ (partagés slides + rapport)
├── presentation/    presentation.tex / .pdf
├── rapport/         rapport.tex / .pdf
├── data/            Oxford-IIIT Pets (téléchargé, non versionné)
├── features/        embeddings en cache (.npz, régénérés par le notebook, non versionnés)
└── references.bib
```

## Reproduire la démonstration

```bash
pip install -r notebook/requirements.txt
# torchvision doit correspondre à la version de torch (ex. CUDA 12.1) :
# pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Données : split trainval d'Oxford-IIIT Pets (miroir Hugging Face) + masques trimaps
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('timm/oxford-iiit-pet', 'data/train-00000-of-00001.parquet', repo_type='dataset', local_dir='data/pets_hf')"
curl -L -o data/annotations.tar.gz https://thor.robots.ox.ac.uk/pets/annotations.tar.gz && tar -xzf data/annotations.tar.gz -C data

# Exécution complète (≈ 10 min la première fois sur un GPU de 4 Go, ≈ 5 min ensuite grâce au cache)
cd notebook && jupyter nbconvert --to notebook --execute --inplace demo_dinov2.ipynb
```

Les poids des modèles (DINOv2-S/B, MAE-B, DeiT-S) sont téléchargés automatiquement depuis le Hub Hugging Face au premier lancement. DINOv2-B est ignoré s'il n'est pas présent dans le cache. Si la connexion est lente, `HF_HUB_DISABLE_XET=1` force le téléchargement HTTP classique.

**Mémoire** : le notebook a besoin d'environ 4 à 5 Go de mémoire virtuelle libre. S'il s'arrête avec `Kernel died`, `OSError 1455` (« fichier de pagination insuffisant ») ou `CUDA error: unknown error`, fermez les applications gourmandes (WSL/Docker, navigateurs, autres notebooks), puis relancez. Les features déjà extraites sont gardées en cache dans `features/` : la seconde exécution prend environ 5 minutes.

**Démo live** : après *Run All*, appeler `search("ma_photo.jpg")`, `search("https://…")` ou `search(image_PIL, model="DeiT-S/16")`.

## Compiler les PDF

```bash
cd presentation && pdflatex presentation && bibtex presentation && pdflatex presentation && pdflatex presentation
cd ../rapport   && pdflatex rapport && bibtex rapport && pdflatex rapport && pdflatex rapport
```

Les valeurs expérimentales citées dans les slides et le rapport proviennent de `figures/results.tex`, écrit par la dernière cellule du notebook. Après une nouvelle exécution, il suffit de recompiler.
