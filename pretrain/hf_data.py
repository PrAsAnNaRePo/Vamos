from PIL import Image
from io import BytesIO
from base64 import b64decode
from datasets import load_dataset
from torch.utils.data import Dataset

class LLavaDataset(Dataset):
    def __init__(self, processor, tokenizer, max_len, dataset_id, dataset_limit, split='train'):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        print("Loading data...")

        self.dataset = load_dataset("MMInstruction/M3IT", dataset_id, split=f'train[:{dataset_limit}%]' if dataset_limit > 0 else 'train' if split == 'train' else 'validation[:25%]')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instruction = self.dataset[index]["instruction"]
        outputs = self.dataset[index]["outputs"]
        conv = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<img></img><|im_end|>\n<|im_start|>assistant\n{outputs}<|im_start|>"
        conv = self.tokenizer(conv, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        image_base64_str_list = self.dataset[index]["image_base64_str"]
        img = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert('RGB')
        img = self.processor(images=img, return_tensors="pt", padding=True)['pixel_values']
        return {"image": img, "input_ids": conv['input_ids'].reshape(-1)}

