# 🎯 Thumbnail Classifier _Based on People Feelings_

> **TL;DR**: Extrage cadre dintr‑un trailer animat, detectează fețe cu YOLOv8‑n și clasifică expresiile folosind un CNN antrenat pe FER‑2013, astfel încât creatorii de conținut să aleagă în câteva secunde cel mai captivant thumbnail.

---

## 🚀 Motivația Proiectului

- **Public‑țintă:** creatori aflați la început de drum, fără buget pentru design grafic.
- **De ce thumbnail‑ul?** Miniatura este primul contact vizual – decide 70‑90 % din rata de click.
- **Economisire timp & bani:** automatizează extragerea și filtrarea emoțiilor în loc să plătești editări manuale.

> 📊 _Fun fact:_ YouTube raportează că schimbarea thumbnail‑ului poate crește CTR‑ul cu până la **30 %**.

---

## 🧩 Logica Proiectului

```mermaid
flowchart LR
    V(Video)-->F[Split în cadre]
    F-->P{Eliminare poze fara fata folosind MediaPipe}
    P-->D[Detectare fețe<br/>YOLOv8‑n]
    D-->E[Crop fețe]
    E --> G [Detectie ploze blurry folosind Laplacian]
    G --> H [Detectie poze blurry folosind FFT]
    H --> I [Detectie poze blurry folosind FFT personalizat]
    I --> J [Detectie poze care nu contin ochii inchisi]
    J --> K [Clasificare emoție<br/>CNN ‑ FER2013]
    K --> L [Selectare cadre „best hit”]
```

---

## 🛠️ Instrumente & Date

| Componentă              | Alegere                 | De ce?                                                                                                                              |
| ----------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Video sursă**         | Trailer animat 🎞️       | Expresii stilizate ⇒ caz de test greu. Dacă reușește aici, pe fețe reale va merge și mai bine.                                      |
| **Detector fețe**       | `yolov8n.pt`            | 3 M param. ⇒ rulează pe CPU/edge, suficient pentru o singură clasă; facut pentru a elimina falsurile pozitive si de a capta fetele. |
| **Dataset detector**    | **WIDER FACE**          | +30 K imagini, diversitate mare.                                                                                                    |
| **Clasificator emoții** | CNN custom, input 48×48 | Pornit de la **FER‑2013**. Dezavantaje: imagini mici, zgomotoase, > dar augmentarea + fine‑tuning compensează.                      |
| **Config YAML**         | `widerface_yolo.yaml`   | Leagă directoarele _train/val_, `nc=1`, `names=['face']`.                                                                           |

---

## ⚙️ Implementare (pe scurt)

<summary><strong>Stack tehnologic</strong></summary>

```text
Python · OpenCV · Ultralytics YOLOv8 · TensorFlow/Keras · Torch · Scikit-learn · Mediapipe · Decord · Cuda
```

---

## 🐞 Provocări

- Găsirea unui dataset potrivit
  Avand in vedere ca am decis sa clasific folosind o retea neurala, nu am avut alta varianta decat fer2013, in ciuda dezavantajelor setului de date.
- Performanța clasificatorului
  Clasificatorul in ciuda faptului ca beneficiaza de data augmentation corespunzator pentru setul de date, este destul de slab din cauza lipsei blocurilor multiple si a calitatii setului de date
- Ambiguitatea sentimentelor din cauza lipsei performantei clasificatorului |

---

## 📚 Ce am învățat

- Fine‑tuning YOLO vs. pre‑training from scratch.
- Cum afectează rezoluția setului de date calitatea predicțiilor
