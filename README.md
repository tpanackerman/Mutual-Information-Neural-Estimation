# Mutual-Information-Neural-Estimation

Ứng dụng mạng neuron trong việc ước lượng thông tin tương hỗ 

## Danh sách thành viên
Trần Phi Anh Nhật - 20234029
Phạm Hồng Duy Minh - 20234025
Hoàng Đức Trung - 20234041
Trần Độ - 20233999

## Giới thiệu đề tài
Mục tiêu của dự án là xây dựng một mô hình Deep Learning để ước lượng lượng Thông tin tương hỗ (Mutual Information - MI) giữa các biến ngẫu nhiên liên tục.
- Phương pháp: Sử dụng mạng Neural để tối ưu hóa cận dưới Donsker-Varadhan (MINE).
- Dữ liệu kiểm chứng: Dữ liệu phân phối Gaussian đa chiều (vì loại dữ liệu này có thể tính được MI chính xác bằng công thức toán học để so sánh).

## Cài đặt
pip install -r setup.txt

## Chạy chương trình
streamlit run main.py
