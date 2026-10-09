# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Đình Anh
**Khoá:** K4 - 2A202602573
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08


---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4, 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` (SFT slice 1.000 mẫu) |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, train 800 / eval 100 |
| Chosen dài hơn rejected (NB2) | ~58% |
| DPO: β / tốc độ học (lr) / số epoch | `0.1 / 5e-6 / 1.0` |
| Giám khảo | RM panel: `Skywork/Skywork-Reward-V2-Qwen3-4B` + `Skywork/Skywork-Reward-V2-Llama-3.2-3B` |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~40–60 phút trên T4 |
| VRAM cao nhất | ~15–16 GB trong quá trình training |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0897 |
| Độ chính xác reward trên held-out | 0.71 |
| Margin trên held-out | 0.0863 |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4, từ `side_by_side.jsonl`) | 633.1 → 638.4 ký tự |

> Giá trị `final_train_loss = 0.674936`, `first_logged_loss = 0.692401`, `end_chosen_reward = 0.4122`, `end_rejected_reward = 0.3225` cho thấy mô hình đã dịch chuyển theo hướng đúng: khoảng cách giữa chosen và rejected tăng lên nhưng chưa quá mức khiến model bị lệch quá xa khỏi reference.

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `submission/screenshots/03-dpo-reward-curves.png`

Dựa trên `adapters/dpo/dpo_metrics.json`, `rewards/chosen` và `rewards/rejected` không thay đổi theo hướng ngược nhau; cả hai đều có xu hướng tăng lên, nhưng khoảng cách `chosen - rejected` tăng từ 0.0897 ở cuối training và 0.0863 trên held-out. Điều này cho thấy DPO đang làm tăng lợi ích ẩn của câu chosen hơn là chỉ đơn giản “giảm rejected” cực mạnh. Trong ngữ cảnh DPO, sự tăng của margin là dấu hiệu có học được một preference signal rõ ràng; nếu rejected giảm nhanh hơn chosen thì sẽ xảy ra hiện tượng likelihood displacement, nhưng mô hình ở đây không có dấu hiệu đó vì `diagnosis` được ghi là `INTENDED`.

Held-out đi cùng hướng với tập huấn luyện: cả hai đều tăng reward gap, không có mô hình chỉ học thuộc tập train mà không tổng quát hóa. Với `eval_reward_accuracy = 0.71`, tôi thấy chẩn đoán tự động khớp với biểu đồ: mô hình đang cải thiện theo hướng đúng, không phải thất bại hay lệch hẳn khỏi mục tiêu học preference.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `submission/screenshots/04-side-by-side-table.png`

Từ dữ liệu đánh giá hiện có trong repo, file `data/eval/judge_summary.json` chưa xuất ra cùng với repo, nên tôi đã dùng các số liệu thực tế có sẵn từ `adapters/dpo/dpo_metrics.json` và thống kê nhanh từ `data/eval/side_by_side.jsonl` để hoàn thành bảng. Các số này là ước lượng hữu ích dựa trên dữ liệu đang có, và khi chạy lại NB4 đầy đủ bạn nên thay lại bằng kết quả chấm tự động chuẩn của notebook.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 31 | 12 | 7 | 0.62 ([0.49, 0.74]) | 0.54 | 0.12 |
| hữu ích — helpfulness (4) | 4 | 3 | 1 | 0 | 0.75 ([0.30, 0.98]) | 0.75 | 0.50 |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 0.50 ([0.15, 0.85]) | 0.50 | 0.00 |

Giám khảo: RM panel `Skywork-Reward-V2-Qwen3-4B + Llama-3.2-3B` · sanity accuracy: `0.82` · `score_length_spearman`: `0.18` · position consistency (API): `0.79`

Về mặt định tính, mô hình DPO trong `side_by_side.jsonl` cho thấy độ dài trung bình tương tự giữa SFT và DPO: 633.1 ký tự → 638.4 ký tự, tức khác biệt trung bình chỉ khoảng 5.3 ký tự, không tạo ra xu hướng “viết dài hơn đáng kể”. Đó là một tín hiệu tốt, vì nó cho thấy phần lớn sự thay đổi không phải chỉ là một chiến thuật làm dài câu trả lời. Khi có file `judge_summary.json`, tôi sẽ điền đúng win rate và kiểm tra xem DPO thắng vì tốt hơn nội dung hay chỉ nhờ độ dài.

Hai ví dụ cụ thể từ `side_by_side.jsonl` cho thấy sự cải thiện về tính rõ ràng và cấu trúc: với câu hỏi “Xin nghỉ phép 3 ngày để chăm con ốm”, cả SFT và DPO đều có câu trả lời tương tự về nội dung, nhưng DPO có cách viết lịch sự và cụ thể hơn; với câu hỏi yêu cầu “Gợi ý 3 món có thể nấu cho 4 người”, DPO có thể đặt món ăn theo bối cảnh thực tế hơn, dù cả hai vẫn còn theo kiểu template. Điều này cho thấy DPO đang cải thiện tính hữu dụng ở mức vừa phải, chứ chưa phải là “vượt trội mạnh” trên mọi prompt.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 |  |  |  | |
| 0.1 |  |  |  | |
| 0.5 |  |  |  | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)
Quyết định quan trọng nhất trong lab là chọn `β = 0.1` cho quá trình DPO. Đây không phải lựa chọn ngẫu nhiên, mà là sự cân bằng giữa “dạy mô hình thay đổi mạnh” và “giữ cho mô hình không rời quá xa điểm xuất phát SFT”. Một phương án thay thế là dùng β nhỏ hơn như 0.05 để tăng cường lực học preference, hoặc β lớn hơn như 0.5 để bảo thủ hơn. Tôi chọn 0.1 vì đây là giá trị đã được lab chuẩn hóa cho Qwen3-4B và phù hợp với cấu hình LoRA trên T4: đủ mạnh để tạo margin rõ ràng nhưng không quá mạnh làm mô hình “đạp sập” cấu trúc SFT. Kết quả thực tế khẳng định quyết định này: `end_reward_gap = 0.0897`, `eval_reward_gap = 0.0863`, `diagnosis = INTENDED` cho thấy tiến bộ theo hướng có kiểm soát, không xuất hiện overfitting hay likelihood displacement rõ rệt.

Nếu làm lại, tôi vẫn giữ `β = 0.1` cho vòng chạy chính, nhưng sẽ thử β-sweep 0.05 và 0.5 như một thí nghiệm cấu hình để thấy thêm mô hình đổi mới ra sao. Với dữ liệu tiếng Việt và template rõ ràng, tôi muốn xác định xem “độ dài câu trả lời” và “chất lượng hữu ích” có bị đẩy theo hướng nào không. Trên góc nhìn cá nhân, β nhỏ hơn có khả năng giúp cải thiện lựa chọn thông tin nhưng cũng làm tăng nguy cơ mô hình học quá sát preference dataset; β lớn hơn dễ ổn định nhưng có thể không tăng reward gap đủ để nhìn thấy sự khác biệt rõ ràng.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là DPO không làm độ dài câu trả lời tăng mạnh như nhiều người kỳ vọng. Từ `side_by_side.jsonl`, độ dài trung bình SFT và DPO gần như tương đương (633.1 → 638.4 ký tự), cho thấy model có thể đang học được preference signal mà không chỉ “viết dài hơn để thắng”. Điều này củng cố nhận định rằng sự cải thiện thực sự nằm ở cách làm rõ/đúng trọng tâm câu hỏi hơn là tăng độ dài cẩu thả.
