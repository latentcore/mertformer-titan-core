# Çıkarım Sözleşmesi

## Girdiler
- **Metin girişi** (UTF-8)
- **Tokenizer**: çalışma zamanında `meta-llama/Llama-3.3-70B-Instruct` üzerinden yüklenir
- **Maksimum bağlam**: 4096 token (`cfg.max_seq_len`)

## Çıktılar
- Token ID'leri ve çözülmüş metin
- Çıktı uzunluğu **çağıran tarafından belirlenir** (örn. chat varsayılanı 128 token)

## Notlar
- Mobil çıkarım için ONNX export sağlanır.
- Sözleşme, üretim koşularından sonra güncellenebilir.
