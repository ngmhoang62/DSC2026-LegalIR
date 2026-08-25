# EXP-022 — E5@100 + BM25@50 union audit

## Mẫu query/gold để đọc

### BM25 cứu, E5@100 bỏ sót

| QID | Query | Gold document | E5 rank | BM25 rank |
|---|---|---|---:|---:|
| 25724 | Mục đích cung cấp thông tin trong quá trình thực hiện hoạt động phải đảm bảo những gì? | 55551 — TCVN 12594 2018 ISO 21103 2014 Du lich mao hiem Thong tin cho nguoi tham gia 918497 | — | 1 |
| 61940 | Chống thất thu đối với doanh nghiệp, cá nhân kinh doanh trong lĩnh vực thương mại, dịch vụ, nhất là chống thất thu thuế trong kinh doanh, chuyển nhượng bất động sản? | 162235 — Thong tu 47 2022 TT BTC huong dan xay dung du toan ngan sach va ke hoach tai chinh 2023... | — | 1 |
| 144970 | Giấy ủy quyền vay vốn có thời hạn 10 năm đã được chứng thực thì được lưu trữ trong thời gian 10 năm hay chỉ 2 năm? | 259656 — Nghi dinh 23 2015 ND CP cap chung thuc ban sao tu ban chinh chung thuc chu ky 266857 | — | 4 |
| 49114 | Đối tượng nào được hưởng chế độ phụ cấp ưu đãi nghề khi hoạt động biểu diễn nghệ thuật trong quân đội? | 166505 — nghi dinh 204 2004 nd cp che do tien luong doi voi can bo cong chuc vien chuc luc luong... | — | 4 |
| 7332 | Bộ Tài chính hướng dẫn thực hiện Thông tư 31/2022/TT-BTC như thế nào?  | 106258 — Cong van 5731 TCHQ TXNK 2022 thuc hien Thong tu 31 2022 TT BTC 548158 | — | 4 |
| 43762 | Việc chẩn đoán bệnh viêm phổi ở trẻ em có dựa trên kết quả kiểm tra cận lâm sàng không? | 234757 — Luat kham benh chua benh nam 2009 98714 | — | 5 |
| 60226 | Cơ quan nào có thẩm quyền giao nhiệm vụ thực hiện xuất bản phẩm sử dụng ngân sách nhà nước? | 150719 — Nghi dinh 32 2019 ND CP dau thau cung cap san pham dich vu cong su dung ngan sach nha n... | — | 5 |
| 24598 | Mẫu chi tiết tài sản dài hạn (tài sản cố định) mới tăng năm 2022? | 185842 — Thong tu 96 2021 TT BTC he thong mau bieu su dung trong cong tac quyet toan 495766 | — | 6 |

### E5 cứu, BM25@50 bỏ sót

| QID | Query | Gold document | E5 rank | BM25 rank |
|---|---|---|---:|---:|
| 145468 | Thuốc thử dùng chẩn đoán bệnh Perkinsus Marinus ở nhuyễn thể hai mảnh nhỏ có những loại nào cần được điều chế? | 125504 — TCVN 8710 10 2015 chuan doan benh do Perkinsus Marinus nhuyen the hai manh nho 915370 | 1 | — |
| 149690 | Người dân vứt xác heo bị dịch tả lợn Châu Phi ra môi trường xung quanh có bị xử lý vi phạm hành chính không?  | 88309 — Nghi dinh 90 2017 ND CP quy dinh xu phat vi pham hanh chinh trong linh vuc thu y 336190 | 1 | — |
| 155776 | Các lỗi phạt nguội thường hay bị phạt đối với xe máy gồm những lỗi nào? | 50885 — Nghi dinh 100 2019 ND CP xu phat vi pham hanh chinh linh vuc giao thong duong bo va duo... | 1 | — |
| 162854 | Khái niệm về sư trụ trì chùa được hiểu như thế nào? Điều kiện để trở thành sư trụ trì là gì? | 44802 — Luat tin nguong ton giao 2016 322934 | 1 | — |
| 163230 | Du học sinh đủ điều kiện được cấp bằng cần làm hồ sơ gì để được tiếp nhận về nước? | 14929 — Nghi dinh 86 2021 ND CP cong dan Viet Nam ra nuoc ngoai hoc tap giang day nghien cuu kh... | 1 | — |
| 22720 | Trường hợp Ngân hàng bán đấu giá tài sản thế chấp để thu hồi nợ, thì bên thế chấp (chủ tài sản) có được đăng ký mua tài sản đó không? | 125793 — Luat dau gia tai san 2016 280115 | 1 | — |
| 22952 | Để trở thành Chủ tịch nước thì ứng viên phải đảm bảo những tiêu chuẩn nào? | 170733 — Quy dinh 214 QD TW 2020 tieu chuan chuc danh can bo thuoc dien Ban Chap hanh Trung uong... | 1 | — |
| 4040 | Trường hợp thực hiện hợp đồng trong đó có ghi giá hợp đồng và thanh toán bằng ngoại hối (USD) thì bị xử phạt như thế nào? | 105483 — Nghi dinh 88 2019 ND CP xu phat vi pham hanh chinh trong linh vuc tien te va ngan hang ... | 1 | — |

### BM25 novel cứu ngoài cả E5 Top-150

| QID | Query | Gold document | E5 rank | BM25 rank |
|---|---|---|---:|---:|
| 25724 | Mục đích cung cấp thông tin trong quá trình thực hiện hoạt động phải đảm bảo những gì? | 55551 — TCVN 12594 2018 ISO 21103 2014 Du lich mao hiem Thong tin cho nguoi tham gia 918497 | — | 1 |
| 144970 | Giấy ủy quyền vay vốn có thời hạn 10 năm đã được chứng thực thì được lưu trữ trong thời gian 10 năm hay chỉ 2 năm? | 259656 — Nghi dinh 23 2015 ND CP cap chung thuc ban sao tu ban chinh chung thuc chu ky 266857 | — | 4 |
| 7332 | Bộ Tài chính hướng dẫn thực hiện Thông tư 31/2022/TT-BTC như thế nào?  | 106258 — Cong van 5731 TCHQ TXNK 2022 thuc hien Thong tu 31 2022 TT BTC 548158 | — | 4 |
| 60226 | Cơ quan nào có thẩm quyền giao nhiệm vụ thực hiện xuất bản phẩm sử dụng ngân sách nhà nước? | 150719 — Nghi dinh 32 2019 ND CP dau thau cung cap san pham dich vu cong su dung ngan sach nha n... | — | 5 |
| 24598 | Mẫu chi tiết tài sản dài hạn (tài sản cố định) mới tăng năm 2022? | 185842 — Thong tu 96 2021 TT BTC he thong mau bieu su dung trong cong tac quyet toan 495766 | — | 6 |
| 37826 | Thiết bị điện tử gia dụng bao gồm những thiết bị nào?  | 214057 — Cong van 3699 TCHQ TXNK 2022 thue gia tri gia tang thiet bi dien tu gia dung nhap khau ... | — | 8 |
| 116230 | Trường hợp khách hàng sử dụng thiết bị khác với thiết bị đăng ký thì Tổ chức tín dụng có giải pháp gì? | 159830 — Cong van 7262 NHNN TT 2022 ung dung du lieu dan cu trong hoat dong ngan hang 536222 | — | 9 |
| 87534 | Quy định về đổi mới, tăng cường, công tác thanh tra, kiểm tra; siết chặt kỷ luật phòng, chống tham nhũng như thế nào? | 266221 — Nghi quyet 18 NQ TW 2022 hoan thien the che su dung dat tao dong luc phat trien thu nha... | — | 10 |

### Cả hai cùng có gold

| QID | Query | Gold document | E5 rank | BM25 rank |
|---|---|---|---:|---:|
| 100126 | Tiêu chuẩn tuyển quân đối với công dân tham gia nghĩa vụ quân sự | 297709 — Thong tu 148 2018 TT BQP quy dinh tuyen chon va goi cong dan nhap ngu 396402 | 1 | 2 |
| 100138 | Để được tham gia xét chọn danh hiệu Doanh nghiệp đổi mới công nghệ tiêu biểu, doanh nghiệp phải đáp ứng những điều kiện nào? | 77697 — Quyet dinh 1244 QD BKHCN 2019 Quy che xet chon ton vinh Doanh nghiep doi moi cong nghe ... | 1 | 1 |
| 100140 | Nghề công tác xã hội được hiểu như thế nào? | 195363 — Thong tu 01 2017 TT BLDTBXH tieu chuan dao duc nghe nghiep nguoi lam cong tac xa hoi 33... | 1 | 2 |
| 100152 | Trình tự lập báo cáo tình hình tài chính nhà nước được thực hiện như thế nào? | 42598 — Thong tu 133 2018 TT BTC huong dan lap Bao cao tai chinh nha nuoc 383300 | 2 | 1 |
| 100158 | Phẫu thuật mở ngực nhỏ tạo dính màng phổi xong thì người bệnh sẽ được theo dõi ra sao? | 202376 — Quyet dinh 5732 QD BYT 2017 ky thuat Ngoai khoa chuyen khoa Phau thuat Tim mach Long ng... | 1 | 1 |
| 100160 | Gia đình người khuyết tật được hưởng học bổng phải chuẩn bị những hồ sơ gì khi theo học tại cơ sở giáo dục công lập? | 245971 — Thong tu lien tich 42 2013 TTLT BGDDT BLDTBXH BTC chinh sach giao duc nguoi khuyet tat ... | 1 | 1 |
| 100176 | Chương trình liên hoan văn nghệ quần chúng được tổ chức ở bao nhiêu cấp? | 169119 — Thong tu 09 2016 TT BVHTTDL to chuc thi lien hoan van nghe quan chung 328074 | 1 | 1 |
| 100242 | Hồ sơ đề nghị cấp giấy phép cho cơ sở sản xuất phương tiện bay siêu nhẹ gồm những gì?  | 276613 — Thong tu 35 2017 TT BQP tieu chuan du dieu kien bay thu tuc cap giay phep thiet ke san ... | 1 | 1 |

### Cả hai cùng bỏ sót gold

| QID | Query | Gold document | E5 rank | BM25 rank |
|---|---|---|---:|---:|
| 100522 | Nhân viên chuyên môn kỹ thuật rà phá bom mìn phải đáp ứng đầy đủ các yêu cầu gì? | 116851 — Thong tu 02 2017 TT BQP hoat dong huan luyen an toan ve sinh lao dong 337143 | — | — |
| 101324 | Xét thăng hạng chức danh nghề nghiệp từ Lưu trữ viên trung cấp lên Lưu trữ viên cần đáp ứng những điều kiện gì?  | 74494 — Luat vien chuc 2010 115271 | — | — |
| 101938 | Chi Cục trưởng thuộc Bộ có quyền hạn cụ thể là gì? | 136014 — Thong tu 12 2022 TT BNV vi tri viec lam cong chuc lanh dao nghiep vu chuyen mon dung ch... | — | — |
| 102004 | Mức thưởng cho Đảng viên được trao tặng huy hiệu Đảng hiện nay? | 43443 — Nghi dinh 38 2019 ND CP muc luong co so doi voi can bo cong chuc vien chuc va luc luong... | — | — |
| 102220 | Cha mẹ có hành vi bạo hành con trai của mình được hiểu như thế nào? | 96450 — Hien phap nam 2013 215627 | — | — |
| 102682 | Xin nghỉ nhưng công ty không đồng ý thì xử lý thế nào? | 176929 — Nghi dinh 24 2018 ND CP giai quyet khieu nai to cao trong linh vuc lao dong 359065 | — | — |
| 102718 | Những nội dung nào liên quan công tác hậu kiểm về an toàn thực phẩm năm 2023? | 42184 — Chi thi 17 CT TTg 2017 tang cuong quan ly nha nuoc chan chinh hoat dong quang cao 348756 | — | — |
| 102928 | Chính sách hỗ trợ đào tạo nghề nghiệp đối với người chấp hành án phạt tù là gì? | 56310 — Thong tu 18 2022 TT BLDTBXH bai bo van ban quy pham phap luat 531025 | — | — |

