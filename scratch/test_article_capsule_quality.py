import json
from pathlib import Path
import sys
import io
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append("d:/Study/DSC2026/LegalIR/src")

tok = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
m = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', torch_dtype=torch.float16).cuda().eval()

# Let's test QID 82848
q = "Có được công chứng Giấy chứng nhận do nước ngoài cấp không?"

# Real relevant article from Luật Công chứng (doc 114415): Điều 78 / công chứng bản dịch giấy tờ tiếng nước ngoài
d_gold_correct_article = """[Văn bản] Luat-Cong-chung-2014
[Cấu trúc] Chương V. THỦ TỤC CÔNG CHỨNG HỢP ĐỒNG, GIAO DỊCH, BẢN DỊCH > Điều 61. Công chứng bản dịch
[Nội dung] Điều 61. Công chứng bản dịch
1. Việc dịch giấy tờ, văn bản từ tiếng Việt sang tiếng nước ngoài hoặc từ tiếng nước ngoài sang tiếng Việt để công chứng phải do người phiên dịch là cộng tác viên của tổ chức hành nghề công chứng thực hiện.
2. Công chứng viên tiếp nhận bản chính giấy tờ, văn bản cần dịch, kiểm tra và giao cho người phiên dịch. Không được công chứng bản dịch trong các trường hợp: Giấy tờ, văn bản được cấp do cơ quan, tổ chức có thẩm quyền của nước ngoài chưa được hợp pháp hóa lãnh sự theo quy định."""

# The bad capsule that was passed before:
d_gold_bad_capsule = """[Văn bản] Luat-Cong-chung-2014
[Phạm vi] Điều 1. Phạm vi điều chỉnh Luật này quy định về công chứng viên, tổ chức hành nghề công chứng...
[Nội dung] Điều 24. Thay đổi nội dung đăng ký hoạt động của Văn phòng công chứng"""

d_false_positive = """[Văn bản] Quyet-dinh-1024-QD-BTP-2018-cong-bo-thu-tuc-hanh-chinh-duoc-sua-doi-linh-vuc-chung-thuc
[Nội dung] quyền của nước ngoài cấp, công chứng hoặc chứng nhận chưa được hợp pháp hóa lãnh sự"""

inp = tok([q, q, q], [d_gold_bad_capsule, d_gold_correct_article, d_false_positive], padding=True, truncation=True, return_tensors='pt').to('cuda')
with torch.inference_mode():
    scores = m(**inp).logits.reshape(-1).tolist()

print("Scores:")
print(f"1. Gold with Bad Capsule (Old): {scores[0]:.4f}")
print(f"2. Gold with Correct Article Chunk: {scores[1]:.4f}")
print(f"3. False Positive Document: {scores[2]:.4f}")
