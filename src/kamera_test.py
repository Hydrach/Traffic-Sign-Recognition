import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🔥 MODEL VE ETIKET AYARLARI (Zaten sende var olan kısımlar)
model_path = "best.onnx"
net = cv2.dnn.readNetFromONNX(model_path)
CLASS_NAMES = ["100_hiz", "120_hiz", "20_hiz", "30_hiz", "50_hiz", "60_hiz", "70_hiz", "80_hiz", "Girilmez", "Kamyonlar_icin_sollama_yasaktır", "Tüm_araclar_icin_sollama_yasaktır", "ana_yol", "ana_yol_tali_yol_kavsagi", "bisiklet_yolu", "calisma_var", "dikkat", "doner_kavsak", "dur", "gizli_buzlanma", "ileri_mecburi_yon", "ileri_ve_saga_mecburi_yon", "ileri_ve_sola_mecburi_yol", "kamyon_giremez", "kasisli_yol", "kaygan_zemin", "okul_gecidi", "saga_mecburi_yon", "saga_tehlikeli_viraj", "sagdan_daralan_yol", "sagdan_gidiniz", "sola_mecburi_yon", "sola_tehlikeli_devamli_viraj", "sola_tehlikeli_viraj", "soldan_gidiniz", "tasit_trafigine_kapali", "vahsi_hayvan_gecebilir", "yaya_gecidi", "yol_ver", "ısıklı_ısaret"]

# '0' yerine tırnak içinde telefonundaki IP adresini yazıyorsun
cap = cv2.VideoCapture(0)

# 🔥 MATPLOTLIB PENCERESİ AYARI
plt.ion() # İnteraktif modu aç (canlı güncelleme için)
fig, ax = plt.subplots()
img_display = None

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 480))

    # Görüntü İşleme ve Tahmin (Önceki kodun aynısı)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224, 224), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    probs = preds[0]
    class_id = np.argmax(probs)
    confidence = float(probs[class_id])

    if confidence > 0.90:
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Bilinmeyen"
        cv2.putText(frame, f"{class_name} %{confidence*100:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (40, 40), (frame.shape[1]-40, frame.shape[2]-40), (0, 255, 0), 2)

    # 🔥 PENCERE GÜNCELLEME (Burası kilit nokta)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV BGR kullanır, Matplotlib RGB ister
    if img_display is None:
        img_display = ax.imshow(frame_rgb)
    else:
        img_display.set_data(frame_rgb)
    
    plt.pause(0.01) # Görüntünün ekrana düşmesi için çok kısa bekle
    plt.draw()

    # Terminalden de durumu izleyelim
    print(f"Canlı İzleme Aktif... Tespit: {class_name if confidence > 0.90 else 'Yok'}")

cap.release()
