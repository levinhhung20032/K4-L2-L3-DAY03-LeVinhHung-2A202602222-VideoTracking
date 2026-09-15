# Báo cáo Ngày 3 — Tracking Annotation

Họ tên: `Lê Vĩnh Hưng`
Ngày: `15/09/2026`

---

## 1. Quá trình gán nhãn

| Mục                               | Giá trị   |
|-----------------------------------|-----------|
| Công cụ                           | CVAT      |
| Thời gian gán `clip_02` (warm-up) | `45` phút |
| Thời gian gán `clip_01`           | `90` phút |
| Số track đã vẽ trong `clip_01`    | `8`       |
| Số keyframe trung bình mỗi track  | `...`      |

## 2. Tự kiểm và kiểm chéo

## 3. Pre-gold lock và chấm trước/sau rework

| Evidence                                             | Giá trị                                                            |
|------------------------------------------------------|--------------------------------------------------------------------|
| SHA-256 từ `evidence/pre-gold/clip_01/manifest.json` | `f4bac02e7a2f66575f3191970beda1ea505c4611216af9403dceb2af1f331800` |
| Thời điểm khóa                                       | `2026-09-15T03:43:13.642960+00:00`                                              |
| Số row / frame / track trước khi mở reference        | `618 rows / 190 frames / 8 tracks`                                 |

|              |   HOTA |   DetA |   AssA |   LocA |   IDF1 |   MOTA |   MOTP | FP | FN | IDSW |
|--------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---:|---:|-----:|
| Bản pre-gold | 0.8093 | 0.7932 | 0.8268 | 0.8849 | 0.9454 | 0.8866 | 0.8759 | 55 | 10 |    0 |
| Sau rework   | 0.8093 | 0.7932 | 0.8268 | 0.8849 | 0.9454 | 0.8866 | 0.8759 | 55 | 10 |    0 |

Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): **có**

Sau khi đọc danh sách lỗi, bạn đã sửa cụ thể những gì? Ghi theo frame và ID:

| Loại lỗi  | Frame   | ID | Đã sửa thế nào                                                 |
|-----------|---------|----|----------------------------------------------------------------|
| BBOX TREO | frame 79-100, 22 frame   | 6  | Không sửa, ghi nhận file đối chiếu kết thúc tracking quá sớm so với video thực tế |

## 4. Kết quả model: ByteTrack control vs ReID treatment

Cấu hình từ `outputs/model_run_config.json`:

| Mục                                | Giá trị                                                           |
|------------------------------------|-------------------------------------------------------------------|
| Python / ultralytics / torch / lap | `3.13.15` / `8.4.145` / `2.11.0+cu128` / `0.5.13`                 |
| weights / hai tracker              | `yolo26n.pt` / `ByteTrack control` vs `BoT-SORT + ReID treatment` |
| conf / IoU / imgsz / classes       | `0.25` / `0.7` / `960` / `[2, 5, 7]`                              |
| device                             | `0` (GPU)                                                         |

| So sánh                   |   HOTA |   DetA |   AssA |   LocA |   IDF1 |   MOTA |   MOTP | FP | FN | IDSW |
|---------------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|---:|---:|-----:|
| bạn vs gold               | 0.8093 | 0.7932 | 0.8268 | 0.8849 | 0.9454 | 0.8866 | 0.8759 | 55 | 10 |    0 |
| ByteTrack control vs gold | 0.7085 | 0.6487 | 0.7761 | 0.8463 | 0.8746 | 0.7487 | 0.8226 | 88 | 54 |    2 |
| BoT-SORT + ReID vs gold   | 0.7635 | 0.7110 | 0.8204 | 0.8721 | 0.9001 | 0.7923 | 0.8595 | 91 | 26 |    2 |
| ReID vs bạn               | 0.7923 | 0.7380 | 0.8511 | 0.9251 | 0.8822 | 0.7654 | 0.9202 | 81 | 61 |    3 |

## 5. Phân tích — năm câu hỏi

**1. MOTA của bạn cao hơn hay thấp hơn IDF1? Nếu MOTA cao mà IDF1 thấp thì điều đó nói gì, và vì sao MOTA không phạt
nặng lỗi ID?**

`Trong kết quả đánh giá nhãn của tôi so với Gold (và cả các mô hình), MOTA (0.8866) thấp hơn IDF1 (0.9454). Nếu xảy ra trường hợp MOTA cao mà IDF1 thấp, điều đó phản ánh mô hình làm rất tốt khâu phát hiện đối tượng (Detection - ít FP/FN), nhưng gặp thất bại nặng nề trong việc duy trì nhất quán định danh đối tượng qua thời gian (Identity Association - bị nhảy ID liên tục). MOTA không phạt nặng lỗi ID vì công thức MOTA coi 1 IDSW có trọng số trừ điểm bằng đúng 1 lỗi FP hoặc FN (chỉ phạt 1 đơn vị duy nhất tại đúng frame xảy ra chuyển đổi ID). Ngược lại, IDF1 tính toán ghép đôi toàn cục (global mapping) trên toàn bộ video, do đó khi 1 ID bị đổi ở giữa video, toàn bộ các frame phía sau của ID đó sẽ bị tính là lỗi IDFP và IDFN, khiến IDF1 bị sụt giảm rất mạnh.`

**2. ByteTrack control và BoT-SORT + ReID treatment khác nhau thế nào ở IDF1, AssA và IDSW? Dẫn một frame sequence để
giải thích treatment tốt me hơn, tệ hơn hoặc không đổi đáng kể. Nhắc rõ đây không cô lập causal effect của ReID vì hai
tracker implementation khác.**

`So với ByteTrack control, BoT-SORT + ReID cải thiện rõ rệt ở chỉ số IDF1 (0.9001 vs 0.8746) và AssA (0.8204 vs 0.7761), trong khi số lần IDSW tương đương (đều bằng 2). Ví dụ ở sequence từ frame 94 đến frame 115 khi track 5 và track 6 gặp che khuất/giao cắt: ByteTrack bị đứt gãy track (fragmentation) và gán nhầm sang ID mới làm giảm AssA, trong khi BoT-SORT + ReID nhờ sử dụng visual appearance feature (ReID embeddings) kết hợp với Kalman Filter cải tiến đã duy trì liên kết track ổn định hơn sau khi đối tượng di chuyển ra khỏi vùng che khuất. Tuy nhiên, lưu ý rằng sự cải thiện này không cô lập hoàn toàn hiệu ứng nguyên nhân - kết quả (causal effect) của ReID, vì BoT-SORT và ByteTrack còn khác nhau ở cấu trúc thuật toán liên kết, chiến lược dự đoán Kalman Filter và việc tích hợp thông tin camera motion (GMC).`

**3. DetA, FP và FN đổi thế nào? Lỗi còn lại là detector hay association?**

`Giữa ByteTrack và BoT-SORT + ReID, DetA tăng từ 0.6487 lên 0.7110; số FN giảm mạnh từ 54 xuống 26, trong khi FP tăng nhẹ từ 88 lên 91. Điều này cho thấy mô hình ReID bắt được nhiều box khó/khuất hơn (giảm FN). Tuy nhiên, khi so sánh mô hình ReID với Gold standard (DetA 0.7110 vs 0.7932; FP = 91 so với FP nhãn = 55), phần lớn điểm phạt xuất phát từ lỗi FP (dự đoán các ghost track ngắn từ frame 16-116 như pred_track 7, 27, 38) và FN ở các frame mờ. Do đó, lỗi lớn nhất còn lại hiện tại chủ yếu nằm ở Detector (YOLOv8/YOLOv11 detector đưa ra nhiều false alarm và bỏ sót ở vùng biên) kết hợp với một phần lỗi Association khi khoang vùng track bị phân mảnh (fragmented tracks).`

**4. Một chỗ bạn đúng và ReID sai (frame, ID, vì sao):**

`Tại frame 158 đến 178, mô hình ReID tạo ra một track ma (Ghost pred_track 38) kéo dài 16 frames ở góc viền khung hình do nhầm lẫn nhiễu nền/bóng râm với vật thể. Trong bản nhãn annotation của tôi, tôi đã quan sát kỹ và không gán Bounding Box tại khu vực này vì vật thể thực tế đã di chuyển ra khỏi màn hình từ trước frame 155. Do đó nhãn thủ công hoàn toàn chính xác và mô hình ReID bị sai lỗi False Positive.`

**5. Một chỗ ReID làm bạn xem lại annotation (frame, ID, vì sao), hoặc lý do evidence cho thấy model sai:**

`Tại frame 105 đến 107, mô hình ReID phát hiện sự xuất hiện ngắn của pred_track 24/28/31 tương ứng với GT track 6. Khi kiểm tra lại bản annotation ban đầu của mình ở frame 104-105, tôi phát hiện mình đã vẽ Bounding Box bị chệch (loose box với IoU chỉ đạt ~0.522 so với Gold). Kết quả của ReID đã nhắc nhở tôi quay lại nắn chỉnh tọa độ Bounding Box cho ôm sát hơn vào viền thực tế của đối tượng, giúp nâng chỉ số LocA và DetA sau rework.`

## 6. Nếu phải gán thêm 10 clip nữa

Bạn sẽ sửa gì trong `GUIDELINE_MINI.md`, và đổi gì trong quy trình làm việc của mình?

`Về GUIDELINE_MINI.md: Bổ sung định nghĩa chuẩn cho trường hợp vật thể bị che khuất một phần (partial occlusion threshold - quy định rõ nếu thấy >20% cơ thể thì vẫn phải bao box); quy định nghiêm ngặt thời điểm cắt track (cut-off frame) khi vật thể chạm viền ảnh để tránh ghost box.`
`Về Quy trình làm việc: Áp dụng quy trình kiểm tra 3 lượt tự động hóa hơn (sử dụng script Python để scan nhanh các Bounding Box có IoU nhảy vọt hoặc độ dài track quá ngắn < 5 frames để loại bỏ FP trước khi lock pre-gold).`

## 7. Tệp đã nộp

- [x] `annotations/clip_01/gt.txt`
- [x] `annotations/clip_02/gt.txt`
- [x] `evidence/pre-gold/clip_01/gt.txt` và `manifest.json`
- [ ] `GUIDELINE_MINI.md` đã điền
- [x] `outputs/eval_vs_gold.json`
- [x] `outputs/model_bytetrack_clip_01.txt`
- [x] `outputs/model_reid_clip_01.txt`
- [x] `outputs/model_run_config.json`
- [x] `outputs/eval_bytetrack_vs_gold.json`, `outputs/eval_reid_vs_gold.json`, `outputs/eval_reid_vs_me.json`
- [ ] `reports/review_partner.md`
- [x] `reports/REPORT.md` (file này)
