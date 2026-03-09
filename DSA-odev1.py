import pickle
import os
import numpy as np
import matplotlib.pyplot as plt



veri_klasoru = "cifar-10-batches-py"


if not os.path.exists(veri_klasoru):
    print("HATA: 'cifar-10-batches-py' klasörü bulunamadı.")
    quit()


# Eğitim verilerini okuyup birleştir
egitim_verileri_listesi = []
egitim_etiketleri_listesi = []

# 5 eğitim batch dosyasını sırayla oku
for batch_numarasi in range(1, 6):
    batch_yolu = os.path.join(veri_klasoru, f"data_batch_{batch_numarasi}")

    with open(batch_yolu, "rb") as dosya:
        batch_sozlugu = pickle.load(dosya, encoding="latin1")

    # data resimlerin piksel verisi
    # labels sınıf numaralarını
    egitim_verileri_listesi.append(batch_sozlugu["data"])
    egitim_etiketleri_listesi.extend(batch_sozlugu["labels"])


# Eğitim verilerini tek büyük dizi haline getir
egitim_verileri = np.vstack(egitim_verileri_listesi)
egitim_etiketleri = np.array(egitim_etiketleri_listesi)


# Test verisini oku
test_yolu = os.path.join(veri_klasoru, "test_batch")

with open(test_yolu, "rb") as dosya:
    test_sozlugu = pickle.load(dosya, encoding="latin1")

test_verileri = test_sozlugu["data"]
test_etiketleri = np.array(test_sozlugu["labels"])


# Sınıf isimlerini oku
meta_yolu = os.path.join(veri_klasoru, "batches.meta")

with open(meta_yolu, "rb") as dosya:
    meta_sozlugu = pickle.load(dosya, encoding="latin1")

sinif_isimleri = meta_sozlugu["label_names"]



print("CIFAR-10 veri seti yüklendi.")
print("Sınıflar:", sinif_isimleri)
print()
print("Kullanılacak algoritmayı seçiniz:")
print("1 = L1/Manhattan")
print("2 = L2/Öklid")


#mesafe türünü al
mesafe_secimi = input("Seçiminiz: ").strip()

while mesafe_secimi not in ["1", "2"]:
    print("Geçersiz seçim yaptınız.")
    mesafe_secimi = input("Lütfen 1 veya 2 giriniz: ").strip()


# K değerini al
k_degeri_yazisi = input("k değerini giriniz: ").strip()

while not k_degeri_yazisi.isdigit() or int(k_degeri_yazisi) <= 0:
    print("Geçersiz k değeri girdiniz.")
    k_degeri_yazisi = input("Lütfen pozitif bir tam sayı giriniz: ").strip()

k_degeri = int(k_degeri_yazisi)


# Kullanıcıdan indeks al
print()
print("Test kümesinden sınıflandırılacak nesneyi seçiniz.")
print("Geçerli indeks aralığı: 0 ile", len(test_verileri) - 1)

nesne_indeksi_yazisi = input("Test nesnesi indeksi: ").strip()

while not nesne_indeksi_yazisi.isdigit() or not (0 <= int(nesne_indeksi_yazisi) < len(test_verileri)):
    print("Geçersiz indeks girdiniz.")
    nesne_indeksi_yazisi = input("Lütfen 0 ile 9999 arasında bir tam sayı giriniz: ").strip()

nesne_indeksi = int(nesne_indeksi_yazisi)


# Seçilen test nesnesini al
test_nesnesi = test_verileri[nesne_indeksi]

# Kontrol içingerçek etiketi sakla
gercek_etiket_numarasi = test_etiketleri[nesne_indeksi]
gercek_etiket_adi = sinif_isimleri[gercek_etiket_numarasi]


# Mesafe hesaplarında taşma olmaması için
egitim_verileri = egitim_verileri.astype(np.float32)
test_nesnesi = test_nesnesi.astype(np.float32)


# L1: |x - y| toplamı
# L2: sqrt((x - y)^2 toplamı)
if mesafe_secimi == "1":
    print()
    print("L1/Manhattan kullanılıyor...")
    farklar = np.abs(egitim_verileri - test_nesnesi)
    uzakliklar = np.sum(farklar, axis=1)
else:
    print()
    print("L2/Öklid Kullanılıyor...")
    farklar = egitim_verileri - test_nesnesi
    kareler = farklar ** 2
    toplam_kareler = np.sum(kareler, axis=1)
    uzakliklar = np.sqrt(toplam_kareler)


# En küçük uzaklıklara sahip örneklerin indekslerini bul
sirali_indeksler = np.argsort(uzakliklar)

# İlk k tanesi en yakın komşuları
en_yakin_komsu_indeksleri = sirali_indeksler[:k_degeri]

# Bu komşuların etiketlerini al
en_yakin_komsu_etiketleri = egitim_etiketleri[en_yakin_komsu_indeksleri]


# Şimdi çoğunluk oylaması yap
etiket_sayilari = np.bincount(en_yakin_komsu_etiketleri, minlength=10)

# En çok görülen etiket tahmin edilen sınıf
tahmin_etiket_numarasi = np.argmax(etiket_sayilari)
tahmin_etiket_adi = sinif_isimleri[tahmin_etiket_numarasi]


# Sonuçlar
print()
print("SONUÇ:")
print("Seçilen test nesnesi indeksi:", nesne_indeksi)
print("k değeri:", k_degeri)
print("Gerçek sınıf:", gercek_etiket_adi)
print("Tahmin edilen sınıf:", tahmin_etiket_adi)
print()

print("En yakın komşuların sınıf dağılımı:")
for sinif_numarasi in range(len(sinif_isimleri)):
    print(f"{sinif_isimleri[sinif_numarasi]}: {etiket_sayilari[sinif_numarasi]}")


# Sınıflandırılan resmi görselleştiren ksısm
resim = test_verileri[nesne_indeksi].reshape(3, 32, 32).transpose(1, 2, 0)

plt.figure(figsize=(4, 4))
plt.imshow(resim)
plt.title(f"Tahmin: {tahmin_etiket_adi} | Gerçek: {gercek_etiket_adi}")
plt.axis("off")
plt.show()