# Thumbnail Classifier Based on People's Feelings

> Extrage cadre dintr-un trailer animat, detectează fețe cu YOLOv8-n și clasifică expresiile folosind un CNN antrenat pe FER-2013, astfel încât creatorii de conținut să aleagă în câteva secunde cel mai captivant thumbnail.

## Cuprins

- [Motivație](#motivație)
- [Stack tehnic & date](#stack-tehnic--date)
- [Instalare & rulare](#instalare--rulare)
- [Structura proiectului](#structura-proiectului)
- [Pipeline pas cu pas](#pipeline-pas-cu-pas)
- [Structura fișierelor de output](#structura-fișierelor-de-output)
- [Realizări tehnice cheie](#realizări-tehnice-cheie)
- [Provocări](#provocări)
- [Ce am învățat](#ce-am-învățat)

## Motivație

- **Public-țintă:** creatori aflați la început de drum, fără buget pentru design grafic.
- **De ce thumbnail-ul?** Miniatura este primul contact vizual - decide 70-90% din rata de click.
- **Economisire timp & bani:** automatizează extragerea și filtrarea emoțiilor în loc să plătești editări manuale.

## Stack tehnic & date

| Componentă | Alegere | De ce? |
| --- | --- | --- |
| Video sursă | Video animat | Expresii stilizate = caz de test greu; dacă funcționează aici, pe fețe reale va merge și mai bine. |
| Detector fețe | `yolov8n.pt` | 3M parametri, rulează pe CPU, suficient pentru o singură clasă și cu puține fals-pozitive. |
| Dataset detector | WIDER FACE | +30K imagini, diversitate mare. |
| Clasificator emoții | CNN from scratch | Simplu, antrenat pe FER-2013; imaginile sunt mici și zgomotoase, dar augmentarea + fine-tuning compensează. |

**Tehnologii:** Python · OpenCV · Ultralytics YOLOv8 · TensorFlow/Keras · PyTorch · Scikit-learn · MediaPipe · Decord · (rulare pe MPS/CUDA/CPU)

## Instalare & rulare

**Cerințe:** Python 3.9+ și pip.

```bash
# 1. Clonează repo-ul
git clone <repo-url>
cd Generate-Thumbnails-Based-On-Feelings

# 2. Creează un mediu virtual și instalează dependințele
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Rulează pipeline-ul complet
python src/main.py
```

Pipeline-ul citește video-ul `The Present.mp4` și populează treptat folderele din `data/` și `output/` descrise mai jos.

## Structura proiectului

```text
.
├── config/ # Configurări dataset (ex: widerface_yolo.yaml)
├── docs/
│ └── documentation.pdf # Documentație tehnică detaliată
├── models/
│ ├── yolov8n.pt # Model YOLOv8 pre-antrenat (bază)
│ └── emotion_recognition.h5 # CNN antrenat pe FER-2013
├── src/
│ ├── main.py # Punct de intrare, rulează pipeline-ul complet
│ ├── predict_emotions.py # Clasificare emoții și triere pe foldere
│ ├── detection/ # Detecție fețe, ochi închiși, blur/motion
│ ├── training/ # Augmentare date și antrenare CNN emoții
│ └── utils/ # Extragere cadre, pregătire YOLO, selectare persoană principală
├── data/ # Cadre extrase, dataset YOLO, rezultate finale pe emoții
├── output/ # Rezultate intermediare ale fiecărei etape de filtrare
└── requirements.txt
```

## Pipeline pas cu pas

Pentru detalii tehnice suplimentare, consultă `docs/documentation.pdf`.

| # | Etapă | Funcție / Fișier | Output |
| --- | --- | --- | --- |
| 1 | Extrage cadre din video, 1/s (Decord + thread pool) | `video_to_frames_one_per_second`, `src/utils/process_frames.py` | `data/test_frames/` |
| 2 | Filtrare inițială cadre cu fețe (MediaPipe) | `detect_faces_mediapipe` / `detect_faces`, `src/detection/face_detection_mediapipe.py` | `output/test_frames_faces/` |
| 3 | Antrenare rapidă YOLOv8 pentru detecție de fețe | `train_yolo_model`, `src/detection/face_detection_yolov8.py` (config: `config/widerface_yolo.yaml`, model bază: `models/yolov8n.pt`) | `data/widerface_yolo/yolo_model/weights/best.pt` |
| 4 | Re-detectare fețe cu YOLOv8 și salvare imagini adnotate | `detect_faces_yolo`, `src/detection/face_detection_yolov8.py` | `output/test_frames_faces_yolo/` |
| 5 | Eliminare cadre cu ochi închiși (EAR, MediaPipe Face Mesh) | `detect_closed_eyes` / `eliminate_closed_eyes`, `src/detection/filter_closed_eyes.py` | `output/test_frames_without_closed_eyes/` |
| 6 | Eliminare cadre cu blur/motion (3 verificări: Laplacian clasic, Laplacian cu kernel personalizat, FFT pe zona feței, prag ~140, `radius=60`) | `eliminate_photos_with_motion`, `src/detection/filter_motion.py` | `output/test_frames_without_motion/` |
| 7 | Selectarea persoanei „principale" (distanța medie a landmark-urilor față de primul cadru) | `detect_most_present_face`, `src/utils/get_main_person.py` | `output/final_frames/` |
| 8 | Clasificare emoții și triere pe foldere (`['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`) | `classify_faces_in_directory`, `src/predict_emotions.py` (clasificator: `models/emotion_recognition.h5`) | `data/emotions/<Emoție>/` |

Pipeline-ul complet, cu toți pașii înlănțuiți, este definit în `src/main.py`.

## Structura fișierelor de output

- `data/test_frames/` - cadre extrase, 1/s
- `output/test_frames_faces/` - cadre cu cel puțin o față (MediaPipe)
- `output/test_frames_faces_yolo/` - cadre adnotate cu bounding box-uri (YOLO)
- `output/test_frames_without_closed_eyes/` - fără ochi închiși
- `output/test_frames_without_motion/` - fără blur/motion
- `output/final_frames/` - cadre finale, persoana principală
- `data/emotions/<Emoție>/` - triere finală după emoție

## Realizări tehnice cheie

- Pipeline cu mai multe etape de filtrare: pre-filtrare MediaPipe, re-detecție YOLO, 3 metode de blur/motion (Laplacian, kernel, FFT pe regiunea feței)
- EAR (Eye Aspect Ratio) pentru a evita cadrele cu ochi închiși - relevant direct pentru calitatea unui thumbnail
- Eficiență I/O: Decord + ThreadPool pentru extragerea rapidă a cadrelor, 1/s
- Cod modular, separat pe responsabilități: `utils/`, `detection/`, `training/`, `predict_emotions.py`
- Antrenare YOLO integrată în flow, configurată pe dataset-ul WIDER FACE
- Clasificator de emoții CNN cu augmentare, early-stopping și `ReduceLROnPlateau`
- Fișiere de output structurate pe emoții, pentru alegerea rapidă a thumbnail-ului

## Provocări

- **Găsirea unui dataset potrivit** - clasificarea folosind o rețea neurală a impus FER-2013, în ciuda dezavantajelor setului de date.
- **Performanța clasificatorului** - deși beneficiază de augmentare corespunzătoare, clasificatorul rămâne limitat din cauza lipsei unor blocuri de rețea mai complexe și a calității modeste a datelor.
- **Ambiguitatea sentimentelor** - o consecință directă a performanței limitate a clasificatorului.

## Ce am învățat

- Fine-tuning YOLO vs. antrenare from scratch.
- Cum afectează rezoluția setului de date calitatea predicțiilor.
