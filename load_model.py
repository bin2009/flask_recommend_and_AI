import torch
import gdown
# Link tải file từ Google Drive (có thể cần chỉnh sửa link thành ID)
url = 'https://drive.google.com/uc?id=1m-2Q4EYxARBobdObFizIICBr1nHJAO55'
output = 'weight_model2/a.pth'
gdown.download(url, output, quiet=False)

from transformers import AutoModel, AutoTokenizer

# Bước 1: Định nghĩa mô hình PhoBERT
class PhoBertModel(torch.nn.Module):
    def __init__(self, phobert):
        super(PhoBertModel, self).__init__()
        self.bert = phobert
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 6)  # Giả sử có 6 lớp

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden_state, output_1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooler = self.pre_classifier(output_1)
        activation_1 = torch.nn.Tanh()(pooler)

        drop = self.dropout(activation_1)
        output_2 = self.classifier(drop)
        output = torch.nn.Sigmoid()(output_2)
        return output


# Bước 2: Tải mô hình PhoBERT và tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")  # Thay bằng tên mô hình của bạn nếu khác
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

model = PhoBertModel(phobert)

# Bước 4: Tải trọng số vào mô hình
# weights_path = './weight_model/phoBertModel_weights_50k_8.pth'  # Đường dẫn tới file trọng số
weights_path = './weight_model2/a.pth' # Đường dẫn tới file trọng số
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()