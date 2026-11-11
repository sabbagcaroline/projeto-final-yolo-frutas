from pathlib import Path
from collections import Counter
import random
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

random.seed(3)
nomes = {0: "apple", 1: "tangerine", 2: "pear", 3: "watermelon", 4: "durian", 5: "lemon",
         6: "grape", 7: "pineapple", 8: "dragon_fruit", 9: "korean_melon", 10: "cantaloupe"}
base = Path("Projeto_Final/dataset")
modelo = YOLO("Projeto_Final/resultados/treino_final/weights/best.pt")
fonte = ImageFont.load_default(size=18)
saida = Path("Projeto_Final/resultados/comparacao")
saida.mkdir(exist_ok=True)


def caixa(d, xyxy, cor, texto, w):
    d.rectangle(xyxy, outline=cor, width=w)
    d.text((xyxy[0] + 3, xyxy[1] + 2), texto, fill=cor, font=fonte)


def preparar(im):
    w, h = im.size
    k = 400 / max(w, h)
    return im.resize((max(1, int(w * k)), max(1, int(h * k)))), k


for i, c in enumerate([0, 10, 8, 5, 1], start=1):
    candidatos = []
    for lab in (base / "labels" / "test").glob("*.txt"):
        linhas = [l for l in lab.read_text().splitlines() if l.strip()]
        classes = [int(l.split()[0]) for l in linhas]
        if c in classes and len(linhas) <= 4:
            candidatos.append(lab)
    lab = random.choice(candidatos)
    img_path = next((base / "images" / "test").glob(lab.stem + ".*"))
    original = Image.open(img_path).convert("RGB")
    base400, k = preparar(original)
    w2, h2 = base400.size

    gt = base400.copy()
    d = ImageDraw.Draw(gt)
    gt_cont = Counter()
    for l in lab.read_text().splitlines():
        p = l.split()
        if len(p) != 5:
            continue
        cl = int(p[0])
        x, y, bw, bh = map(float, p[1:])
        caixa(d, [(x - bw / 2) * w2, (y - bh / 2) * h2, (x + bw / 2) * w2, (y + bh / 2) * h2], "green", nomes[cl], 3)
        gt_cont[nomes[cl]] += 1

    pr = base400.copy()
    d = ImageDraw.Draw(pr)
    pr_cont = Counter()
    r = modelo.predict(str(img_path), conf=0.25, imgsz=416, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = [v * k for v in b.xyxy[0].tolist()]
        cl = int(b.cls[0])
        s = float(b.conf[0])
        caixa(d, [x1, y1, x2, y2], "red", nomes[cl] + " " + format(s, ".2f"), 3)
        pr_cont[nomes[cl]] += 1

    folha = Image.new("RGB", (1200, 430), "white")
    paineis = [(base400, "Original"), (gt, "Anotacao (verde)"), (pr, "Predicao conf=0.25 (vermelho)")]
    for j, (im, t) in enumerate(paineis):
        folha.paste(im, (j * 400, 30))
        ImageDraw.Draw(folha).text((j * 400 + 6, 5), t, fill="black", font=fonte)
    folha.save(saida / ("comparacao_" + str(i) + ".jpg"))
    print(i, img_path.name, "| anotados:", dict(gt_cont), "| previstos:", dict(pr_cont))

print("pronto: pasta comparacao")