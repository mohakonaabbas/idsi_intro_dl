# IDSI_intro_dl
Ce dépôt contient les cours et TP pour le cours d'introduction à l'IDSI sur l'apprentissage profond commencé en 2025

# Tutoriel Pratique PyTorch
## Introduction
Ce README sert de guide complet pour la session pratique PyTorch (TP) sur les fondamentaux des réseaux de neurones et l'optimisation. Il suppose des connaissances de base en Python et un accès à une machine avec GPU NVIDIA (si disponible). Les étudiants implémenteront et expérimenteront avec les concepts clés pour comprendre leur impact sur les métriques de performance comme la vitesse, l'utilisation de la mémoire et la précision. Temps estimé : 4-6 heures, avec des exemples de code en notebooks Jupyter ou scripts Python. Utilisez TensorBoard pour visualiser les résultats.

## Prérequis
- Python 3.8+ installé.
- Familiarité de base avec NumPy et l'algèbre linéaire.
- Accès à un environnement avec GPU (recommandé mais non obligatoire ; repli sur CPU fourni).
- Installer les bibliothèques requises via pip : `https://pytorch.org/get-started/locally/` et `pip install matplotlib scikit-learn tensorboard` pour la visualisation et les métriques.


# Pourquoi apprendre Pytorch ?

* Momentum dans la recherche
https://trends.google.com/trends/explore?date=all&geo=FR&q=pytorch,keras&hl=fr

* Pytorch est très performant

![Performance of frameworks](https://unfoldai.com/storage/2024/08/keras-pytorch-performance.jpg)


## Pourquoi PyTorch pourrait être préféré à Keras ou Tensorflow : Une comparaison

**PyTorch** et **Keras** sont deux bibliothèques Python populaires pour le deep learning, chacune avec ses forces et ses faiblesses. Choisir l'une ou l'autre dépend souvent des besoins spécifiques du projet. Voici quelques raisons pour lesquelles un développeur pourrait préférer PyTorch à Keras :

### 1. **Flexibilité et contrôle:**
* **Graphiques dynamiques:** PyTorch offre une grande flexibilité grâce à ses graphiques dynamiques, permettant de modifier le modèle à la volée. Cela est particulièrement utile pour la recherche et les modèles expérimentaux.
* **Bas niveau:** PyTorch est plus proche du matériel, ce qui donne un meilleur contrôle sur l'optimisation et le débogage.
* **Intégration avec d'autres outils:** PyTorch s'intègre facilement avec d'autres bibliothèques Python, ce qui le rend très polyvalent.

### 2. **Communauté et écosystème:**
* **Recherche active:** PyTorch est très populaire dans la communauté de recherche en apprentissage profond, ce qui signifie que de nouvelles fonctionnalités et améliorations sont souvent ajoutées.
* **Grand écosystème:** PyTorch dispose d'un écosystème riche et en constante évolution, avec de nombreux outils et bibliothèques complémentaires.

### 3. **Performances:**
* **Tensor opérations:** PyTorch offre des performances élevées grâce à son optimisation des opérations sur les tenseurs.
* **GPU accélération:** PyTorch est bien intégré avec les GPU, ce qui est essentiel pour les modèles de deep learning exigeants.

### 4. **Pythonic:**
* **Naturel:** PyTorch est conçu pour être très Pythonic, ce qui facilite l'apprentissage et l'utilisation pour les développeurs Python expérimentés.
* 

## Qu'est-ce que TensorBoard et Comment le Configurer ?
TensorBoard est un outil de visualisation open-source développé par l'équipe TensorFlow, mais pleinement compatible avec PyTorch. Il permet de suivre et d'analyser les expériences d'apprentissage automatique en temps réel, en affichant des graphiques interactifs pour les métriques comme la perte (loss), la précision (accuracy), les histogrammes de poids et de gradients, les courbes de convergence, ainsi que le graphe du modèle. C'est particulièrement utile pour diagnostiquer des problèmes comme le surapprentissage, les gradients qui s'annulent ou explosent, et pour comparer différentes configurations d'optimiseurs ou d'activations.

### Avantages pour ce TP
- Visualisez les courbes de perte et de précision par époque pour observer la convergence.
- Comparez les métriques (vitesse, mémoire) entre axes via des tableaux et graphiques.
- Affichez le graphe du modèle pour comprendre l'architecture.
- Suivez l'évolution des gradients pour analyser les problèmes de vanishing/exploding.
- Partagez des visualisations avec les étudiants via un navigateur web.

### Installation
TensorBoard est installé via `pip install tensorboard` (déjà inclus dans les prérequis). Une fois installé, lancez-le depuis le terminal avec :
```bash
tensorboard --logdir=runs
```
Ouvrez ensuite `http://localhost:6006` dans votre navigateur pour voir le tableau de bord. Les logs sont stockés dans un dossier `runs/` (créé automatiquement).

### Configuration de Base en Code
Pour intégrer TensorBoard dans votre code PyTorch, importez `SummaryWriter` et créez un writer au début de votre script ou notebook. Voici un exemple simple à ajouter avant l'étape 1 :
```python
from torch.utils.tensorboard import SummaryWriter

# Créez un writer pour cet exercice (dossier runs sera créé)
writer = SummaryWriter('runs/pytorch_tp_intro')

# Exemple : Loggez une métrique simple au step 0
writer.add_scalar('Métriques/Version_PyTorch', torch.__version__, 0)

# Pour fermer à la fin : writer.close()
```
Dans les boucles d'entraînement (voir code starter), ajoutez des logs comme :
```python
writer.add_scalar('Loss/Entraînement', loss.item(), global_step=epoch * len(train_loader) + batch_idx)
writer.add_scalar('Accuracy/Test', test_acc, global_step=epoch)
writer.add_histogram('Poids/Couche1', model.fc1.weight, global_step=epoch)  # Histogrammes des poids
# Pour le graphe du modèle (une fois) :
writer.add_graph(model, sample_data)  # sample_data = next(iter(train_loader))[0]
```
À la fin de l'expérience, appelez `writer.close()` pour sauvegarder. Consultez les logs en temps réel pour analyser les axes du TP.


## Étape 1 : Découverte PyTorch et Configuration GPU
Commencez par explorer les bases de PyTorch pour assurer un démarrage en douceur. Après la configuration GPU, initialisez TensorBoard comme indiqué ci-dessus.

### Installation
1. Suivez le guide d'installation officiel PyTorch pour votre système (CPU ou GPU). Utilisez la commande ci-dessus pour le support GPU.
2. Vérifiez l'installation : Exécutez `import torch; print(torch.__version__)` dans un shell Python pour confirmer la version (par exemple, 2.1+).

### Test GPU avec nvidia-smi
1. Ouvrez un terminal et exécutez `nvidia-smi` pour vérifier la disponibilité du GPU, la version du pilote et la mémoire. Cet outil affiche l'utilisation du GPU en temps réel, ce qui sera utile pour le monitoring pendant l'entraînement.
2. En Python, testez l'accessibilité du GPU :
   ```python
   import torch
   if torch.cuda.is_available():
       print(f"GPU disponible : {torch.cuda.get_device_name(0)}")
       device = torch.device("cuda")
   else:
       print("Utilisation du CPU")
       device = torch.device("cpu")
   ```
   Résultat attendu : Confirme la configuration CUDA si un GPU est détecté ; sinon, repasse au CPU pour tous les exercices.

Cette étape assure la préparation du matériel et introduit les opérations sur tenseurs sur GPU pour des calculs plus rapides. Loggez la disponibilité GPU dans TensorBoard : `writer.add_text('Hardware/GPU', f'{torch.cuda.is_available()}', 0)`.

## Étape 2 : Construction d'un Réseau de Neurones Simple (Axe 1)
Implémentez un réseau de neurones feedforward de base pour une tâche de classification, comme la reconnaissance de chiffres MNIST, et définissez les métriques d'évaluation. Utilisez TensorBoard pour logger la perte et l'accuracy pendant l'entraînement.

### Configuration du Dataset et du Modèle
1. Chargez MNIST : Utilisez `torchvision.datasets.MNIST` pour télécharger et préparer le dataset (images en niveaux de gris 28x28, 10 classes).
   ```python
   import torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
   train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   ```
2. Définissez un réseau simple : 784 neurones d'entrée (image aplatie), une couche cachée (128 neurones), couche de sortie (10 neurones).
   ```python
   import torch.nn as nn

   class SimpleNN(nn.Module):
       def __init__(self):
           super().__init__()
           TODO

       def forward(self, x):
           TODO

   model = SimpleNN().to(device)
   ```

### Boucle d'Entraînement
1. Entraînez pendant 5-10 époques en utilisant l'optimiseur SGD (lr=0.01). Ajoutez des logs TensorBoard dans la boucle.
   ```python
   TODO
   # Exemple de log : writer.add_scalar('Loss/Axe1', loss.item(), epoch)
   ```

### Définition des Métriques
Évaluez la performance du modèle en termes de vitesse, mémoire et précision. Implémentez ces métriques dans une fonction `evaluate_model`.
- **Vitesse** : Mesurez le temps d'entraînement par époque en utilisant `time.time()`. 
- **Utilisation de la Mémoire** : Surveillez la mémoire GPU maximale avec `torch.cuda.max_memory_allocated()` (en octets ; divisez par 1e9 pour GB). Réinitialisez avec `torch.cuda.reset_peak_memory_stats()`.
- **Précision (Accuracy)** : Calculez la précision de classification sur un ensemble de validation (utilisez `test_dataset` de MNIST).

Exécutez le modèle et enregistrez les métriques dans un fichier ou TensorBoard pour les comparaisons dans les axes suivants. Loggez-les : `writer.add_scalars('Métriques/Axe1', {'vitesse': time_s, 'mémoire': mem_gb, 'précision': acc}, epoch)`.

Cet axe introduit les composants principaux de PyTorch : modèles, chargeurs de données, perte et optimiseurs.

## Étape 3 : Test des Fonctions d'Activation (Axe 2)
Modifiez le réseau simple pour expérimenter avec différentes activations et observer les impacts sur les métriques. Utilisez TensorBoard pour comparer les courbes de convergence entre ReLU, Tanh et Sigmoid.

1. Mettez à jour le modèle pour supporter des activations interchangeables : Ajoutez un paramètre dans `__init__` comme `self.activation = activation_fn` (par exemple, `nn.ReLU()`, `nn.Tanh()`, `nn.Sigmoid()`).
   ```python
   # Dans forward : x = self.activation(self.fc1(x))
   ```
2. Entraînez trois variantes : ReLU (par défaut), Tanh, Sigmoid. Utilisez les mêmes hyperparamètres et dataset.
3. Comparez les métriques : Réexécutez l'entraînement pour chacune, en enregistrant la vitesse (les activations comme Sigmoid sont plus lourdes en calcul), la mémoire (similaire, mais les gradients diffèrent), et la précision (Tanh souvent meilleure que Sigmoid pour les gradients qui s'annulent ; ReLU la plus rapide). Loggez par tag : `writer.add_scalar('Loss/ReLU', loss, epoch)`.

Visualisez les courbes de perte avec Matplotlib pour montrer les différences de convergence. Préférez TensorBoard pour des vues interactives multiples.

## Étape 4 : Test des Différents Optimiseurs (Axe 3)
Explorez comment les optimiseurs affectent la dynamique d'entraînement sur le réseau simple. Comparez les runs dans TensorBoard sous l'onglet Scalars.

1. Testez SGD (base), SGD avec Momentum (0.9), Adam (lr=0.001), RMSprop (lr=0.01).

2. Entraînez chacun pendant 10 époques, en réinitialisant les poids du modèle à chaque fois (utilisez `torch.save/load_state_dict` pour la cohérence).
3. Comparaison des métriques :
   - Vitesse
   - Mémoire
   - Précision
Utilisez la même fonction `evaluate_model` et tabulez les résultats (par exemple, précision finale, temps moyen par époque). Loggez : `writer.add_scalars('Accuracy/Optimiseurs', {'SGD': acc_sgd, 'Adam': acc_adam}, epoch)`.

Discutez du rôle du momentum pour accélérer SGD en amortissant les oscillations.

## Étape 5 : Régularisation avec Dropout (Axe 4)
Introduisez la prévention du surapprentissage en utilisant Dropout sur le réseau simple. Utilisez TensorBoard pour tracer les courbes train/val et détecter le surapprentissage.

1. Ajoutez une couche Dropout : Insérez `nn.Dropout(p=0.5)` après l'activation de la couche cachée dans le modèle.
   ```python
   # Dans __init__ : self.dropout = nn.Dropout(0.5)
   # Dans forward : x = self.dropout(torch.relu(self.fc1(x)))
   ```
2. Entraînez avec/sans Dropout (utilisez un ensemble de validation pour diviser train/val).
3. Métriques :
   - Vitesse
   - Mémoire
   - Précision
   - 
Surveillez le surapprentissage via les courbes de perte train vs. val. Loggez : `writer.add_scalars('Loss/Dropout', {'Train': train_loss, 'Val': val_loss}, epoch)`.

## Étape 6 : Ajustement des Hyperparamètres (Axe 5)
Ajustez systématiquement l'architecture du réseau et la dynamique d'apprentissage. Utilisez TensorBoard pour comparer les runs de grid search (créez des sous-dossiers comme `runs/tuning_lr0.01_layers2`).

1. Paramètres à varier :
   - Nombre de couches
   - Neurones par couche
   - Taux d'apprentissage
   - Planification du LR : pour faire décroître le LR
2. Implémentez une recherche en grille
3. Métriques : Suivez les trois

La validation croisée peut-elle aider ici ? Oui, pour une évaluation plus robuste ; loggez les scores CV dans TensorBoard.

## Étape 7 : Normalisation par Lots et Initialisation (Axe 6)
Testez des techniques pour un entraînement stable dans les réseaux plus profonds. Visualisez les histogrammes de gradients dans TensorBoard pour observer vanishing/exploding.

1. Batch Norm : Ajoutez `nn.BatchNorm1d(xx)` après les couches linéaires (avant activation).
   ```python
   # Dans le modèle : self.bn1 = nn.BatchNorm1d(128); x = self.bn1(self.fc1(x))
   ```
2. Stratégies d'initialisation :
   - Initialisation à zéro : `nn.init.zeros_(self.fc1.weight)` 
   - Kaiming (He) : `nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')` 
   - Uniforme par défaut (baseline PyTorch).
3. Entraînez des variantes sur un réseau à 3 couches :
   - Comparez au modèle de base
   - Métriques
   - Vitesse
   - 
Tracez les courbes d'entraînement pour visualiser les problèmes de flux de gradients. Loggez : `writer.add_histogram('Gradients/Couche1', model.fc1.weight.grad, epoch)`.

## Conclusion et Extensions
Les étudiants devraient compiler toutes les expériences dans un rapport comparant les métriques à travers les axes (utilisez des tableaux pour des vues côte à côte). Exportez les visualisations de TensorBoard ou capturez des screenshots pour le rapport. Extensions : Ajoutez des logs pour des datasets plus complexes comme CIFAR-10.
