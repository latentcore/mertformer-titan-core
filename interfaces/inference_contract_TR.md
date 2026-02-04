# Cikartim Sozlesmesi

## Girdiler
- **Metin girisi** (UTF-8)
- **Tokenizer**: calisma zamaninda `meta-llama/Llama-3.3-70B-Instruct` uzerinden yuklenir
- **Maksimum baglam**: 4096 token (`cfg.max_seq_len`)

## Ciktilar
- Token ID'leri ve cozulmus metin
- Cikti uzunlugu **cagiran tarafindan belirlenir** (orn. chat varsayilani 128 token)

## Notlar
- Mobil cikartim icin ONNX export saglanir.
- Sozlesme, uretim kosularindan sonra guncellenebilir.
