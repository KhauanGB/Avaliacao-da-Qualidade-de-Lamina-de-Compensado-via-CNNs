# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÃO DE SEED (REPRODUTIBILIDADE = 42)
# ==============================================================================
import os
import random
import json
import copy
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- DEFINIÇÃO RIGOROSA DA SEED 42 ---
SEED = 42

def fixar_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixar_seeds(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executando pipeline no dispositivo: {device}")

# Parâmetros Globais
DIRETORIO_BASE = '/home/andre/Documents/madeira/Base_Original'
CLASSES = ['A', 'B', 'C', 'D', 'E']
DIRETORIO_SAIDA = './resultados_experimentos'
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# ==============================================================================
# 2. CHAVES DE ATIVAÇÃO DOS EXPERIMENTOS (ESTRUTURA EXPLICITA)
# ==============================================================================
EXPERIMENTOS_PARA_RODAR = {
    # Extratores Tradicionais (Sem Data Augmentation)
    "hist_rgb": False,
    "hist_hsv": False,
    "haralick_glcm": False,
    "lbp_config1": False,
    "lbp_config2": False,
    "hog": False,
    "sift": False,
    
    # Redes Neurais Fim a Fim (Sem Data Augmentation)
    "effnet_b7_tl": False,
    "effnet_b7_scratch": False,
    "effnet_b5_tl": False,
    "effnet_b5_scratch": False,
    "vgg16_tl": False,
    "vgg16_scratch": False,
    "resnet50_tl": False,
    "resnet50_scratch": False,
    
    # Extratores Tradicionais + Data Augmentation
    "hist_rgb_da": True,
    "hist_hsv_da": True,
    "haralick_glcm_da": True,
    "lbp_config1_da": True,
    "lbp_config2_da": True,
    "hog_da": True,
    "sift_da": True,
    
    # Redes Neurais Fim a Fim + Data Augmentation
    "effnet_b7_tl_da": True,
    "effnet_b7_scratch_da": True,
    "effnet_b5_tl_da": True,
    "effnet_b5_scratch_da": True,
    "vgg16_tl_da": True,
    "vgg16_scratch_da": True,
    "resnet50_tl_da": True,
    "resnet50_scratch_da": True,

    # Embeddings das CNNs (Se ativado, extrai das CNNs selecionadas acima)
    "embeddings_cnn": True
}

# ==============================================================================
# 3. GERENCIAMENTO DE CHECKPOINTS E SALVAMENTO INDIVIDUAL
# ==============================================================================
def experimento_ja_concluido(nome_exp):
    nome_arquivo = f"exp_{nome_exp.replace(' ', '_').replace('/', '_')}.json"
    caminho_completo = os.path.join(DIRETORIO_SAIDA, nome_arquivo)
    return os.path.exists(caminho_completo)

def salvar_resultado_individual(nome_exp, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Matriz de Confusão: {nome_exp}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(DIRETORIO_SAIDA, f'cm_{nome_exp.replace(" ", "_").replace("/", "_")}.png'))
    plt.close()

    res = {
        'Experimento': nome_exp,
        'Acuracia': float(acc),
        'Precisao_Weighted': float(prec),
        'Recall_Weighted': float(rec),
        'F1_Score_Weighted': float(f1),
        'Matriz_Confusao': cm.tolist()
    }

    nome_arquivo = f"exp_{nome_exp.replace(' ', '_').replace('/', '_')}.json"
    caminho_json = os.path.join(DIRETORIO_SAIDA, nome_arquivo)
    with open(caminho_json, 'w') as f:
        json.dump(res, f, indent=4)

    print(f"[CONCLUÍDO E SALVO] -> {nome_exp} | Acurácia: {acc*100:.2f}%")

# ==============================================================================
# 4. LEITURA E SPLIT ESTRATIFICADO DO DATASET (SEED 42)
# ==============================================================================
def carregar_dataset(diretorio_base, classes):
    caminhos_imagens = []
    labels = []
    mapa_classes = {nome: idx for idx, nome in enumerate(classes)}

    for nome_classe in classes:
        caminho_classe = os.path.join(diretorio_base, nome_classe)
        if not os.path.exists(caminho_classe): continue

        arquivos = [f for f in os.listdir(caminho_classe) if not f.startswith('.')]
        for nome_img in arquivos:
            caminhos_imagens.append(os.path.join(caminho_classe, nome_img))
            labels.append(mapa_classes[nome_classe])

    return caminhos_imagens, np.array(labels)

caminhos_imgs, y_all = carregar_dataset(DIRETORIO_BASE, CLASSES)
print(f"Dataset carregado: {len(caminhos_imgs)} imagens.")

train_paths, test_paths, y_train, y_test = train_test_split(
    caminhos_imgs, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

# ==============================================================================
# 5. PIPELINES DE DATA AUGMENTATION E CARREGAMENTO
# ==============================================================================
transform_eval = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train_aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Função para aplicar Data Augmentation diretamente nas imagens para métodos tradicionais
def aplicar_aug_imagem_tradicional(caminho):
    img = cv2.imread(caminho)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Rotação / Espelhamento Aleatório
    if random.random() > 0.5:
        img = cv2.flip(img, 1) # Flip Horizontal
    if random.random() > 0.5:
        img = cv2.flip(img, 0) # Flip Vertical
    return img

# ==============================================================================
# 6. EXTRATORES TRADICIONAIS
# ==============================================================================
def extrair_hist_rgb(caminho, img_mat=None, bins=32):
    img = img_mat if img_mat is not None else cv2.imread(caminho)
    if img is None: return None
    img = cv2.resize(img, (128, 128))
    b, g, r = cv2.split(img)
    hist_r = cv2.calcHist([r], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [bins], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [bins], [0, 256])
    hist = np.concatenate([hist_r, hist_g, hist_b]).ravel()
    return hist / (hist.sum() + 1e-7)

def extrair_hist_hsv(caminho, img_mat=None, bins=32):
    img = img_mat if img_mat is not None else cv2.imread(caminho)
    if img is None: return None
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hist_h = cv2.calcHist([h], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [bins], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).ravel()
    return hist / (hist.sum() + 1e-7)

def extrair_haralick(caminho, img_mat=None, distancias=[1, 3], angulos=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    if img_mat is not None:
        img = cv2.cvtColor(img_mat, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (128, 128))
    img_quantized = (img // 8).astype(np.uint8)
    glcm = graycomatrix(img_quantized, distances=distancias, angles=angulos, levels=32, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    vetor_haralick = [np.mean(graycoprops(glcm, p)) for p in props]
    return np.array(vetor_haralick)

def extrair_lbp(caminho, img_mat=None, P=8, R=1):
    if img_mat is not None:
        img = cv2.cvtColor(img_mat, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (128, 128))
    h, w = img.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            centro = img[i, j]
            code = 0
            code |= (img[i-1, j-1] >= centro) << 7
            code |= (img[i-1, j]   >= centro) << 6
            code |= (img[i-1, j+1] >= centro) << 5
            code |= (img[i,   j+1] >= centro) << 4
            code |= (img[i+1, j+1] >= centro) << 3
            code |= (img[i+1, j]   >= centro) << 2
            code |= (img[i+1, j-1] >= centro) << 1
            code |= (img[i,   j-1] >= centro) << 0
            lbp[i-1, j-1] = code
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist.astype("float") / (hist.sum() + 1e-7)

def extrair_hog(caminho, img_mat=None):
    if img_mat is not None:
        img = cv2.cvtColor(img_mat, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (64, 128))
    hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    return hog.compute(img).ravel()

sift_extractor = cv2.SIFT_create()
def extrair_sift_des(caminho, img_mat=None):
    if img_mat is not None:
        img = cv2.cvtColor(img_mat, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (256, 256))
    _, des = sift_extractor.detectAndCompute(img, None)
    return des

def executar_tradicional(X_tr, y_tr, X_te, y_te, prefixo):
    exp_svm = f"{prefixo} + SVM"
    if not experimento_ja_concluido(exp_svm):
        print(f"Executando {exp_svm}...")
        svm = SVC(kernel='rbf', C=1.0, random_state=SEED)
        svm.fit(X_tr, y_tr)
        preds_svm = svm.predict(X_te)
        salvar_resultado_individual(exp_svm, y_te, preds_svm)
    else:
        print(f"[PULADO] Experimento já concluído: {exp_svm}")

    exp_nb = f"{prefixo} + Naive Bayes"
    if not experimento_ja_concluido(exp_nb):
        print(f"Executando {exp_nb}...")
        nb = GaussianNB()
        nb.fit(X_tr, y_tr)
        preds_nb = nb.predict(X_te)
        salvar_resultado_individual(exp_nb, y_te, preds_nb)
    else:
        print(f"[PULADO] Experimento já concluído: {exp_nb}")

def extrair_features_dataset(fn_extracao, usar_da=False):
    if usar_da:
        X_tr = np.array([fn_extracao(p, img_mat=aplicar_aug_imagem_tradicional(p)) for p in train_paths])
    else:
        X_tr = np.array([fn_extracao(p) for p in train_paths])
    X_te = np.array([fn_extracao(p) for p in test_paths])
    return X_tr, X_te

# ==============================================================================
# 7. EXECUTOR DOS MÉTODOS TRADICIONAIS
# ==============================================================================
print("\n" + "="*50 + "\n INICIANDO / RETOMANDO EXPERIMENTOS TRADICIONAIS\n" + "="*50)

tradicionais_map = [
    ("hist_rgb", "Hist RGB", extrair_hist_rgb),
    ("hist_hsv", "Hist HSV", extrair_hist_hsv),
    ("haralick_glcm", "Haralick (GLCM)", extrair_haralick),
    ("lbp_config1", "LBP Config 1", lambda p, img_mat=None: extrair_lbp(p, img_mat, P=8, R=1)),
    ("lbp_config2", "LBP Config 2", lambda p, img_mat=None: extrair_lbp(p, img_mat, P=16, R=2)),
    ("hog", "HOG", extrair_hog)
]

for chave_base, nome_base, fn in tradicionais_map:
    # Sem Data Augmentation
    if EXPERIMENTOS_PARA_RODAR.get(chave_base, False):
        X_tr, X_te = extrair_features_dataset(fn, usar_da=False)
        executar_tradicional(X_tr, y_train, X_te, y_test, nome_base)

    # Com Data Augmentation
    chave_da = f"{chave_base}_da"
    if EXPERIMENTOS_PARA_RODAR.get(chave_da, False):
        X_tr, X_te = extrair_features_dataset(fn, usar_da=True)
        executar_tradicional(X_tr, y_train, X_te, y_test, f"{nome_base} (DA)")

# Tratamento especial para SIFT / BoVW
for chave, usar_da, sufixo in [("sift", False, ""), ("sift_da", True, " (DA)")]:
    if EXPERIMENTOS_PARA_RODAR.get(chave, False):
        nome_exp = f"SIFT (BoVW){sufixo}"
        if not (experimento_ja_concluido(f"{nome_exp} + SVM") and experimento_ja_concluido(f"{nome_exp} + Naive Bayes")):
            des_list = []
            for p in train_paths:
                img_m = aplicar_aug_imagem_tradicional(p) if usar_da else None
                d = extrair_sift_des(p, img_mat=img_m)
                if d is not None: des_list.append(d)
                
            all_des = np.vstack(des_list)
            kmeans = KMeans(n_clusters=100, random_state=SEED, n_init=5)
            kmeans.fit(all_des)

            def construir_hist_sift(path, is_train=False):
                img_m = aplicar_aug_imagem_tradicional(path) if (usar_da and is_train) else None
                d = extrair_sift_des(path, img_mat=img_m)
                hist = np.zeros(100)
                if d is not None:
                    words = kmeans.predict(d)
                    for w in words: hist[w] += 1
                return hist / (hist.sum() + 1e-7)

            X_tr_sift = np.array([construir_hist_sift(p, is_train=True) for p in train_paths])
            X_te_sift = np.array([construir_hist_sift(p, is_train=False) for p in test_paths])
            executar_tradicional(X_tr_sift, y_train, X_te_sift, y_test, nome_exp)

# ==============================================================================
# 8. REDES NEURAIS (DEEP LEARNING) COM EARLY STOPPING E EMBEDDINGS
# ==============================================================================
class DatasetMadeira(Dataset):
    def __init__(self, caminhos, labels, transform=None):
        self.caminhos = caminhos
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        img = cv2.imread(self.caminhos[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def obter_modelo_cnn(nome_modelo, pretrained=True, num_classes=5):
    weights = 'DEFAULT' if pretrained else None
    if nome_modelo == 'efficientnet_b7':
        model = models.efficientnet_b7(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif nome_modelo == 'efficientnet_b5':
        model = models.efficientnet_b5(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif nome_modelo == 'vgg16':
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif nome_modelo == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def treinar_e_avaliar_cnn(nome_exp, nome_modelo, pretrained=True, usar_da=False, max_epochs=100, patience=5):
    if experimento_ja_concluido(nome_exp):
        print(f"[PULADO] Experimento já concluído: {nome_exp}")
        return

    aug_str = "COM Data Augmentation" if usar_da else "SEM Data Augmentation"
    print(f"\n--> Treinando CNN: {nome_exp} ({aug_str} | Até {max_epochs} épocas, Early Stopping = {patience})...")
    fixar_seeds(SEED)

    train_paths_tr, train_paths_val, y_tr, y_val = train_test_split(
        train_paths, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    t_train = transform_train_aug if usar_da else transform_eval

    train_loader = DataLoader(DatasetMadeira(train_paths_tr, y_tr, transform=t_train), batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(DatasetMadeira(train_paths_val, y_val, transform=transform_eval), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(DatasetMadeira(test_paths, y_test, transform=transform_eval), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = obter_modelo_cnn(nome_modelo, pretrained=pretrained, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)

        val_loss = val_loss / len(train_paths_val)
        
        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    [Early Stopping] Parado na época {epoch+1}. Melhor Loss de Validação: {best_loss:.4f}")
                break

    model.load_state_dict(best_model_wts)
    model.eval()
    todas_preds, todas_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            todas_preds.extend(preds.cpu().numpy())
            todas_labels.extend(lbls.numpy())

    salvar_resultado_individual(nome_exp, todas_labels, todas_preds)

def extrair_embeddings_cnn(nome_modelo, pretrained=True, paths=None):
    model = obter_modelo_cnn(nome_modelo, pretrained=pretrained, num_classes=len(CLASSES))
    
    if 'efficientnet' in nome_modelo:
        model.classifier[1] = nn.Identity()
    elif nome_modelo == 'vgg16':
        model.classifier[6] = nn.Identity()
    elif nome_modelo == 'resnet50':
        model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    loader = DataLoader(DatasetMadeira(paths, np.zeros(len(paths)), transform=transform_eval), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    embeddings = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            embeddings.append(feats.cpu().numpy())

    return np.vstack(embeddings)

# ==============================================================================
# 9. EXECUTOR DAS CNNs E EMBEDDINGS
# ==============================================================================
print("\n" + "="*50 + "\n INICIANDO / RETOMANDO EXPERIMENTOS DE DEEP LEARNING\n" + "="*50)

cnns_map = [
    ("effnet_b7_tl", "EfficientNet B7 Com TL", "efficientnet_b7", True),
    ("effnet_b7_scratch", "EfficientNet B7 Sem TL", "efficientnet_b7", False),
    ("effnet_b5_tl", "EfficientNet B5 Com TL", "efficientnet_b5", True),
    ("effnet_b5_scratch", "EfficientNet B5 Sem TL", "efficientnet_b5", False),
    ("vgg16_tl", "VGG16 Com TL", "vgg16", True),
    ("vgg16_scratch", "VGG16 Sem TL", "vgg16", False),
    ("resnet50_tl", "ResNet50 Com TL", "resnet50", True),
    ("resnet50_scratch", "ResNet50 Sem TL", "resnet50", False)
]

for chave_base, nome_base, modelo_id, pretrained in cnns_map:
    # Testa as duas variações: Sem DA e Com DA
    variacoes = [
        (chave_base, nome_base, False),
        (f"{chave_base}_da", f"{nome_base} (DA)", True)
    ]
    
    for chave, nome_exp, usar_da in variacoes:
        if EXPERIMENTOS_PARA_RODAR.get(chave, False):
            # 1. Treino Fim-a-Fim
            treinar_e_avaliar_cnn(nome_exp, modelo_id, pretrained=pretrained, usar_da=usar_da, max_epochs=100, patience=5)

            # 2. Embeddings + SVM e Naive Bayes (se ativado globalmente)
            if EXPERIMENTOS_PARA_RODAR.get("embeddings_cnn", False):
                prefixo_emb = f"Embeddings {nome_exp}"
                if not (experimento_ja_concluido(f"{prefixo_emb} + SVM") and experimento_ja_concluido(f"{prefixo_emb} + Naive Bayes")):
                    print(f"\n--> Extraindo Embeddings para {prefixo_emb}...")
                    X_tr_emb = extrair_embeddings_cnn(modelo_id, pretrained=pretrained, paths=train_paths)
                    X_te_emb = extrair_embeddings_cnn(modelo_id, pretrained=pretrained, paths=test_paths)
                    executar_tradicional(X_tr_emb, y_train, X_te_emb, y_test, prefixo_emb)

# ==============================================================================
# 10. CONSOLIDAÇÃO DOS RESULTADOS GERADOS
# ==============================================================================
print("\n" + "="*50 + "\n COMPILANDO TABELA RESUMO FINAL\n" + "="*50)

lista_json = [f for f in os.listdir(DIRETORIO_SAIDA) if f.startswith('exp_') and f.endswith('.json')]
dados_consolidados = []

for arq in lista_json:
    caminho = os.path.join(DIRETORIO_SAIDA, arq)
    with open(caminho, 'r') as f:
        dados_consolidados.append(json.load(f))

if len(dados_consolidados) > 0:
    df_resumo = pd.DataFrame(dados_consolidados)
    df_resumo = df_resumo.sort_values(by="Acuracia", ascending=False)
    
    df_resumo.to_csv(os.path.join(DIRETORIO_SAIDA, 'resultado_consolidado_geral.csv'), index=False)
    
    print("\nTabela Resumo Atualizada:")
    print(df_resumo[['Experimento', 'Acuracia', 'Precisao_Weighted', 'Recall_Weighted']].to_string(index=False))
else:
    print("Nenhum resultado processado ainda.")