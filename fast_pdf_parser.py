import pdfplumber
from rapidocr_onnxruntime import RapidOCR
import numpy as np

def calculate_intersection_ratio(text_bbox, cell_bbox):
    """
    Tính toán tỷ lệ diện tích giao cắt của text_bbox so với toàn bộ diện tích của text_bbox.
    text_bbox: (x0, y0, x1, y1) - Tọa độ hộp văn bản từ OCR
    cell_bbox: (x0, top, x1, bottom) - Tọa độ ô lưới từ pdfplumber
    """
    t_x0, t_y0, t_x1, t_y1 = text_bbox
    c_x0, c_y0, c_x1, c_y1 = cell_bbox

    # Tọa độ vùng giao cắt
    i_x0 = max(t_x0, c_x0)
    i_y0 = max(t_y0, c_y0)
    i_x1 = min(t_x1, c_x1)
    i_y1 = min(t_y1, c_y1)

    inter_width = max(0, i_x1 - i_x0)
    inter_height = max(0, i_y1 - i_y0)
    inter_area = inter_width * inter_height

    text_area = (t_x1 - t_x0) * (t_y1 - t_y0)
    
    if text_area == 0:
        return 0
    return inter_area / text_area

def parse_pdf_bounding_box_intersection(pdf_path: str, output_txt_path: str):
    # Khởi tạo RapidOCR (Mặc định sẽ tải model đa ngôn ngữ, hỗ trợ rất tốt tiếng Nhật/Anh)
    ocr_engine = RapidOCR()
    
    full_text_output = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            full_text_output.append(f"--- PAGE {page_num + 1} ---")
            
            # BƯỚC 1: Quét chữ một lần duy nhất với RapidOCR
            # Render trang PDF thành ảnh (resolution=72 để 1 pixel = 1 point, giúp đồng bộ tọa độ dễ dàng)
            pil_image = page.to_image(resolution=72).original
            img_array = np.array(pil_image)
            
            ocr_result, _ = ocr_engine(img_array)
            ocr_boxes = []
            
            if ocr_result:
                for box in ocr_result:
                    coords, text, confidence = box
                    # coords là list 4 điểm [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # Chuyển về bounding box dạng (x0, y0, x1, y1)
                    xs = [pt[0] for pt in coords]
                    ys = [pt[1] for pt in coords]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    
                    ocr_boxes.append({
                        "bbox": bbox,
                        "text": text,
                        "used_in_table": False
                    })

            # BƯỚC 2: Bóc tách cấu trúc lưới vật lý bằng pdfplumber
            tables = page.find_tables()
            table_structures = []
            
            for table in tables:
                # pdfplumber trả về thuộc tính .cells là danh sách các hàng, mỗi hàng chứa tọa độ các ô
                table_structures.append(table.cells)

            # BƯỚC 3: Lắp ráp (CPU Intersection)
            page_content_blocks = [] # Lưu trữ cả Text và Table theo thứ tự Y
            
            for table_idx, cells_in_rows in enumerate(table_structures):
                markdown_table = []
                # Xác định tọa độ Y trên cùng của bảng để lát nữa sắp xếp luồng văn bản
                table_top_y = cells_in_rows[0][0][1] if cells_in_rows and cells_in_rows[0] and cells_in_rows[0][0] else 0
                
                for row in cells_in_rows:
                    row_data = []
                    for cell_bbox in row:
                        if cell_bbox is None:
                            row_data.append("")
                            continue
                            
                        cell_text = []
                        for ocr_box in ocr_boxes:
                            if not ocr_box["used_in_table"]:
                                # Kiểm tra xem text có nằm lọt trong ô (giao cắt > 50%)
                                ratio = calculate_intersection_ratio(ocr_box["bbox"], cell_bbox)
                                if ratio > 0.5:
                                    cell_text.append(ocr_box["text"])
                                    ocr_box["used_in_table"] = True
                        
                        # Nối các text trong cùng 1 ô bằng khoảng trắng
                        row_data.append(" ".join(cell_text).replace("|", "\\|").replace("\n", " "))
                    
                    # Cấu trúc thành markdown row
                    markdown_table.append("| " + " | ".join(row_data) + " |")
                    
                    # Thêm dải phân cách header cho bảng (nếu là dòng đầu tiên)
                    if len(markdown_table) == 1:
                        header_sep = "|-" + "-|-".join(["-" * len(col) for col in row_data]) + "-|"
                        markdown_table.append(header_sep)
                
                page_content_blocks.append({
                    "type": "table",
                    "y_coord": table_top_y,
                    "content": "\n".join(markdown_table)
                })

            # BƯỚC 4: Xử lý phần văn bản ngoài bảng (Paragraph)
            non_table_boxes = [box for box in ocr_boxes if not box["used_in_table"]]
            
            # Nhóm các dòng có tọa độ Y gần nhau (sai số 10 points) để tạo thành từng dòng (line)
            # Sau đó sắp xếp theo Y (từ trên xuống), và X (từ trái qua phải)
            for box in non_table_boxes:
                page_content_blocks.append({
                    "type": "text",
                    "y_coord": box["bbox"][1], # Lấy y0 làm mốc
                    "x_coord": box["bbox"][0], # Lấy x0 làm mốc
                    "content": box["text"]
                })
            
            # Sắp xếp lại toàn bộ khối (bảng và text ngoài) theo chiều dọc trang tài liệu
            # Nếu tọa độ Y chênh lệch dưới 10 point, coi như cùng 1 hàng ngang, lúc đó xếp theo chiều ngang X
            page_content_blocks.sort(key=lambda item: (round(item['y_coord'] / 10), item.get('x_coord', 0)))

            # Đẩy vào kết quả cuối
            for block in page_content_blocks:
                full_text_output.append(block["content"])
                
            full_text_output.append("\n") # Ngắt trang

    # Ghi kết quả ra file TXT
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text_output))
        
    print(f"[Hoàn tất] Đã trích xuất và lưu kết quả vào: {output_txt_path}")

# ================= CÁCH SỬ DỤNG =================
if __name__ == "__main__":
    # Bạn thay đổi đường dẫn file input và output tại đây
    input_pdf = r"C:/CS_Major/Contests_2025/DevDay_AI_Competion/Docs/Public/01_masked_758cbc89/07】_masked_2bafa275/【_masked_18fbf841/446f77f1ada144cb871a065cb02e1b1c.pdf" 
    output_txt = "extracted_output.txt"
    
    # Kích hoạt hàm
    parse_pdf_bounding_box_intersection(input_pdf, output_txt)