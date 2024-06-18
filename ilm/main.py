import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import tqdm
import numpy as np
import os

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(128*3, 128*3*3, bias=False)
        self.out = nn.Linear(128*3, 128*3, bias=False)
        self.num_heads = 6
        self.rope = LlamaRotaryEmbedding(128*3 // self.num_heads, max_position_embeddings=4096, base=10000)
    
    def reshape_heads(self, x: Tensor):
        N, T, HID = x.shape
        return x.view(N, T, self.num_heads, -1).permute(0, 2, 1, 3)
    
    def forward(self, x):
        qkv = self.qkv(x) # type: Tensor
        q, k, v = qkv.chunk(3, -1)
        N, T, HID = q.shape
        
        q = self.reshape_heads(q)
        k = self.reshape_heads(k)
        v = self.reshape_heads(v)
        
        position_ids = torch.arange(0, T, device=q.device)
        cos, sin = self.rope(v, position_ids[None, :,])
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        context = F.scaled_dot_product_attention(
            q, k, v, 
            None, 
            dropout_p=0.1 if self.training else 0.0, 
            is_causal=True, 
            scale=1/(q.shape[-1] ** 0.5)
        )
        context = context.permute(0, 2, 1, 3).view(N, T, HID)
        out = self.out(context)
        return out

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(128*3, 128*3*4)
        self.b = nn.Linear(128*3*4, 128*3)
    
    def forward(self, x):
        return self.b(F.gelu(self.a(x)))

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128*3)
        self.attn = Attention()
        self.mlp = MLP()
    
    def forward(self, x: Tensor):
        t = self.norm(x)
        x = x + self.attn(t)
        x = x + self.mlp(t)
        return x

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.r = nn.Embedding(256*2, 128)
        self.g = nn.Embedding(256*2, 128)
        self.b = nn.Embedding(256*2, 128)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        assert input_ids.ndim == 3
        assert input_ids.shape[-1] == 3
        
        r = self.r(input_ids[:, :, 0])
        g = self.g(input_ids[:, :, 1])
        b = self.b(input_ids[:, :, 2])
        
        return torch.cat([r, g, b], dim=-1)

class ILM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = Embedding()
        self.layers = nn.Sequential(*[
            DecoderLayer()
            for _ in range(12)
        ])
        self.proj_r = nn.Linear(128*3, 512)
        self.proj_g = nn.Linear(128*3, 512)
        self.proj_b = nn.Linear(128*3, 512)
    
    def forward(self, input_ids: Tensor, return_logits: bool = False):
        x = self.embedding(input_ids)
        x = self.layers(x)
        
        r = self.proj_r(x)
        g = self.proj_g(x)
        b = self.proj_b(x)

        rgb = torch.stack([r, g, b], dim=-2)
        
        if return_logits:
            return rgb
        
        label = input_ids.view(-1).clone()
        label[:-1] =  label[1:].clone()
        label[-1] = -100
        loss = F.cross_entropy(rgb.view(-1, 512), label)
        
        return loss

def convert_pil_to_vec(img):
    img = img.resize((32, 32))
    
    img = [
        torch.tensor([[256, 256, 256]], dtype=torch.long), # BOS
        torch.tensor(np.array(img.resize((1, 1)).convert('RGB'))).view(-1, 3).to(torch.long),
        torch.tensor(np.array(img.resize((2, 2)).convert('RGB'))).view(-1, 3).to(torch.long),
        torch.tensor(np.array(img.resize((4, 4)).convert('RGB'))).view(-1, 3).to(torch.long),
        torch.tensor(np.array(img.resize((8, 8)).convert('RGB'))).view(-1, 3).to(torch.long),
        torch.tensor(np.array(img.resize((16, 16)).convert('RGB'))).view(-1, 3).to(torch.long),
        torch.tensor(np.array(img.resize((32, 32)).convert('RGB'))).view(-1, 3).to(torch.long),
    ]
    
    img = torch.cat(img, dim=0)
    
    return img

import os
from PIL import Image

def dataaug_pil_to_vec(img: Image.Image):
    imgs = [
        convert_pil_to_vec(img),
        convert_pil_to_vec(img.transpose(Image.TRANSPOSE)),
        convert_pil_to_vec(img.transpose(Image.TRANSVERSE)),
        convert_pil_to_vec(img.transpose(Image.FLIP_LEFT_RIGHT)),
        convert_pil_to_vec(img.transpose(Image.FLIP_TOP_BOTTOM)),
        convert_pil_to_vec(img.transpose(Image.ROTATE_90)),
        convert_pil_to_vec(img.transpose(Image.ROTATE_270)),
        convert_pil_to_vec(img.transpose(Image.ROTATE_180)),
    ]
    return imgs

def load_image_dir(path):
    imgs = []
    for name in os.listdir(path):
        filepath = os.path.join(path, name)
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if any([filepath.lower().endswith(ext.lower()) for ext in exts]):
            imgs.append(Image.open(filepath))
    return imgs

def get_cifar():
    import torchvision
    ds = torchvision.datasets.CIFAR10('./.cache/cifar10', download=True)
    imgs = []
    for img, _ in tqdm.tqdm(ds, desc='cifar10'):
        imgs += dataaug_pil_to_vec(img)
    
    for img in tqdm.tqdm(load_image_dir('./data/tiny_imnet/tiny-imagenet-200/test/images'), desc='imnet test'):
        imgs += dataaug_pil_to_vec(img)
    
    for img in tqdm.tqdm(load_image_dir('./data/tiny_imnet/tiny-imagenet-200/val/images'), desc='imnet valid'):
        imgs += dataaug_pil_to_vec(img)
    
    traindir = './data/tiny_imnet/tiny-imagenet-200/train'
    for subdir in tqdm.tqdm(os.listdir(traindir), desc='imnet train'):
        for img in load_image_dir(os.path.join(traindir, subdir, 'images')):
            imgs += dataaug_pil_to_vec(img)
    
    return torch.stack(imgs, dim=0)

class Trainer:
    def __init__(self):
        self.device = 0
        # N, T, 3
        self.image_data = get_cifar().to(self.device)
        self.train_data = self.image_data[:-1000]
        self.valid_data = self.image_data[-1000:]
        self.model = ILM().to(self.device)
        self.batch_size = 32
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self):
        self.model.train()
        data_index = torch.randperm(len(self.train_data), device=self.train_data.device)
        loss_sum = 0
        loss_count = 0
        with tqdm.tqdm(range(len(self.train_data) // self.batch_size)) as pbar:
            for i in pbar:
                batch = self.train_data[data_index[i * self.batch_size: (i + 1) * self.batch_size]]
                with torch.autocast('cuda', torch.float16):
                    loss = self.model(batch)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss_sum += loss.item()
                loss_count += 1
                pbar.set_description(f'L: {loss_sum / loss_count:.6f}')
    
    def eval_epoch(self):
        self.model.eval()
        with torch.inference_mode():
            loss = self.model(self.valid_data).item()
            print(f'valid loss: {loss}, PPL: {math.exp(loss)}')
    
    def save(self):
        torch.save({
            'model': self.model.state_dict(),
        }, './saves.pth')
    
    def load(self, path: str):
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model'])
        del state
    
    def main(self):
        while True:
            self.train_epoch()
            self.eval_epoch()
            self.save()
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, type=str)
    args = parser.parse_args()
    
    t = Trainer()
    if args.checkpoint is not None:
        t.load(args.checkpoint)
    t.main()