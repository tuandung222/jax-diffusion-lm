"""
================================================================================
MÔ-ĐUN TOKENIZER: CHARACTER-LEVEL TOKENIZER CHO DIFFUSION LM
================================================================================
Triển khai bộ mã hóa cấp độ ký tự (Character-level Tokenizer) độc lập, không dùng
các thư viện ngoài như HuggingFace Tokenizers hay Tiktoken.

VAI TRÒ CỦA TOKENIZER TRONG MÔ HÌNH DIFFUSION LANGUAGE MODEL:
------------------------------------------------------------
Khác với các mô hình sinh tự hồi quy (Autoregressive - AR như GPT) chỉ cần sinh tiếp
token kế tiếp từ trái sang phải, mô hình Discrete Diffusion LM (như MDLM, D3PM)
sử dụng cơ chế "Khuếch tán Rời rạc" với trạng thái hấp thụ (Absorbing State):

1. Token đặc biệt `<mask>` đóng vai trò là "Nhiễu Cực Đại" (Pure Noise / Absorbing State).
   Ở thời điểm t=1.0, toàn bộ chuỗi văn bản là các token `<mask`>.
2. Mô hình học cách khôi phục lại ký tự nguyên bản từ các vị trí đang bị che bởi `<mask`>.
3. Token `<pad>` được sử dụng để căn chỉnh độ dài chuỗi trong một lô (batch).
4. Token `<bos>` và `<eos>` đánh dấu điểm bắt đầu và kết thúc chuỗi văn bản.
5. Token `<unk>` đại diện cho các ký tự nằm ngoài từ vựng.

ƯU ĐIỂM KHI DÙNG CHARACTER TOKENIZER CHO MỤC ĐÍCH GIẢNG DẠY:
-----------------------------------------------------------
- Kích thước từ vựng nhỏ gọn (khoảng 100 - 150 tokens thay vì 32,000 - 128,000 như BPE).
- Giúp ma trận đầu ra và tính toán Softmax cực kỳ nhẹ, có thể huấn luyện và thử nghiệm
  ngay trên CPU hoặc GPU tích hợp của Apple M4 mà không lo tràn VRAM.
- Trực quan hóa quá trình khử nhiễu (Denoising) từng bước sinh chữ cực kỳ rõ ràng!
"""

from typing import List, Dict, Optional, Any, Union


class CharTokenizer:
    """
    Bộ mã hóa cấp độ ký tự hỗ trợ các token đặc biệt phục vụ Diffusion Language Modeling.
    """
    
    PAD_TOKEN = "<pad>"
    MASK_TOKEN = "<mask>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, texts: Optional[List[str]] = None):
        """
        Khởi tạo từ vựng từ danh sách văn bản mẫu hoặc từ tập ký tự ASCII in được.
        
        Args:
            texts: Danh sách các chuỗi văn bản huấn luyện để xây dựng tập ký tự độc nhất.
                   Nếu None, mặc định sử dụng bảng mã chuẩn ASCII in được.
        """
        # Danh sách token đặc biệt luôn được ưu tiên đặt ở đầu bảng mã
        self.special_tokens = [
            self.PAD_TOKEN,
            self.MASK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]
        
        if texts:
            # Thu thập toàn bộ ký tự độc nhất xuất hiện trong dữ liệu huấn luyện
            unique_chars = sorted(list(set("".join(texts))))
        else:
            # Mặc định: Tập ký tự in được chuẩn ASCII (từ mã 32 đến 126) + xuống dòng, thụt lề
            unique_chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
            
        # Ghép các token đặc biệt và các ký tự thường thành từ vựng hoàn chỉnh
        all_tokens = self.special_tokens + [c for c in unique_chars if c not in self.special_tokens]
        
        # Tạo từ điển ánh xạ hai chiều: Ký tự -> ID và ID -> Ký tự
        self.char_to_id: Dict[str, int] = {c: i for i, c in enumerate(all_tokens)}
        self.id_to_char: Dict[int, str] = {i: c for i, c in enumerate(all_tokens)}
        
        # Lưu trữ sẵn chỉ số (ID) của các token đặc biệt để truy xuất tức thời O(1)
        self.pad_id = self.char_to_id[self.PAD_TOKEN]
        self.mask_id = self.char_to_id[self.MASK_TOKEN]
        self.bos_id = self.char_to_id[self.BOS_TOKEN]
        self.eos_id = self.char_to_id[self.EOS_TOKEN]
        self.unk_id = self.char_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Kích thước từ vựng (tổng số lượng token)."""
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Chuyển đổi chuỗi văn bản đầu vào thành danh sách các token ID số nguyên.
        Các ký tự chưa từng gặp trong từ vựng sẽ được thay thế bằng unk_id.
        """
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, token_ids: Any) -> str:
        """
        Giải mã danh sách các token ID trở lại chuỗi ký tự ban đầu.
        Chấp nhận cả List[int], numpy array hoặc JAX DeviceArray.
        
        LƯU Ý TRỰC QUAN HÓA:
        - Các vị trí còn mang token `<mask`> sẽ được hiển thị bằng ký tự khối vuông '█'
          để người dùng dễ dàng quan sát quá trình unmasking từng bước của Diffusion.
        - Các token kiểm soát như `<pad>`, `<bos>`, `<eos>` sẽ được ẩn đi để văn bản rõ ràng.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        chars = []
        for tid in token_ids:
            tid_int = int(tid)
            char = self.id_to_char.get(tid_int, "")
            if char == self.MASK_TOKEN:
                chars.append("█")  # Biểu diễn trực quan cho token còn đang bị che
            elif char not in self.special_tokens:
                chars.append(char)
        return "".join(chars)
