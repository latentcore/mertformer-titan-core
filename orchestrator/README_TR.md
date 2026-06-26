# `orchestrator/` — kapsam dışı, 45K'da inert

English: [README.md](README.md)

**Kapsam sınırı.** Bu paket bir araştırma/ajan iskeletidir. **Kanonik ön-eğitim / 45K koşusunun kapsamı dışındadır** ve **eğitim yolunda inerttir**: `train/train.py` tarafından çalıştırılmaz ve 45K koşusunun eğittiği modele **hiç parametre katmaz**. Yalnızca olası, ayrı bir gelecek aşama için tutulmaktadır.

**Yetenek iddiası yok.** Buradaki hiçbir şey benchmark-doğrulanmış, üretime-hazır veya kanıt-uygun değildir. Ajan/AGI-yönündeki yüzeyler closure matrisinde açıkça "out-of-scope pending" olarak listelenmiştir (bkz. `reports/closure_57_matrix.md` ve kök `README.md`'den referans verilen `out_of_scope_pending_ids` satırları).

**Neden ağaçta duruyor.** Kaldırılması 45K-sonrası temizliğe ertelendi (Pass 4 mühürlü — "no Pass 5"). O zamana kadar, reviewer'ların bunu eğitilen modelin parçası sanmaması için burada kapsam-dışı olarak belgelenmiştir.
