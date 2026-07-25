# Gussian

**Tài liệu cho gussian**

Tày mọi - opacity:
    Trong 3D Gaussian Splatting (3DGS), opacity (độ mờ) là một tham số quyết định khả năng tái tạo các bề mặt sắc nét cũng như các vùng bán trong suốt của mô hình 3D.

    Đưa raw opacity qua Sigmoid để chuẩn hóa về 0-1, vì bản chất 3DGS vẫn là ứng dụng của Gaussian distribution nên cần đưa về phạm vi phân phối chuẩn mới nắc được.

scale_factor: 
    điều chỉnh thông số kích thước một ma trận hiệp phương sai xác định dương (positive-definite) [là ma trận đối xứng có tất cả các trị riêng (eigenvalues) đều lớn hơn 0]. Cụ thể là scale 3 kích thước XYZ
percent_dense: 
    Trong 3D Gaussian Splatting (3DGS), percent_dense là một siêu tham số (hyperparameter) đóng vai trò làm ngưỡng kích thước để thuật toán quyết định cách tăng số lượng (densification) cho các quả cầu Gaussian. 
    clone: chép thêm 1 quả tương tự đặt cạnh nó
    splip: gojo 
    bé hơn percent_dense: clone
    lớn hơn percent_dense: splip
