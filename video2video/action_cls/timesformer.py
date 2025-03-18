#### Fusion v3
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Optional
from transformers import BertTokenizer, BertModel, CLIPVisionModel, TimesformerModel
from .utils import *

class TimeSformerMutimodal(nn.Module):
    def __init__(self, 
                 class_embed =None, 
                 num_frames: int = 30,
                 num_classes: Optional[int] = None,
                 fusion: str = 'transformer',
                 residual: Optional[str] = None,
                 recurrent: Optional[str] = None,
                 video_model: str = "facebook/timesformer-base-finetuned-k400",
                 text_model_name: str = 'bert-base-uncased',
                 visual_text_model: str = "openai/clip-vit-base-patch16"):
        super().__init__()
        assert fusion in {'transformer', 'self_attn', 'cross_attn'}
        assert text_model_name in {'bert-base-uncased', 'FacebookAI/roberta-base', 'FacebookAI/xlm-roberta-base'}
        self.class_embed = class_embed
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.residual = residual
        self.recurrent = recurrent
        # models
        self.fusion = fusion
        self.video_model = video_model
        self.text_model_name = text_model_name
        self.visual_text_model = visual_text_model
        
        if self.class_embed is None:
            self.num_classes, self.embed_dim = (18, 512)
        #### Model initializer
        # timesformer
        self.backbone = TimesformerModel.from_pretrained(self.video_model, num_frames = self.num_frames, ignore_mismatched_sizes= True)
        
        # text model
        self.text_model = BertModel.from_pretrained(self.text_model_name)
        self.text_tokenizer = BertTokenizer.from_pretrained(self.text_model_name)
        
        # visual_text
        self.image_model = CLIPVisionModel.from_pretrained(self.visual_text_model)
                 
        #### BRANCH v1
        # Hai cái này sẽ bổ sung thông tin ngữ nghĩa (residual) sau khi backbone đã làm đủ trò
        self.linear1 = nn.Linear(self.backbone.config.hidden_size, self.embed_dim, bias= True)
        self.temperal_pool = nn.AdaptiveAvgPool1d(self.num_frames)
        # Add thêm positional encoding nữa
        self.pos_encod = PositionalEncoding(self.embed_dim)
        self.video_embedding = nn.Linear(self.embed_dim, self.embed_dim, bias = True)
        
        
        if self.fusion == 'self_attn':
            self.video_self_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
            self.text_self_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        elif self.fusion == 'cross_attn':
            self.cross_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        elif self.fusion == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model= self.embed_dim, 
                nhead=4, 
                batch_first=True
            )

            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        #### BRANCH v2
        self.linear2 = nn.Linear(in_features=self.backbone.config.hidden_size + self.embed_dim, out_features=self.embed_dim, bias=False)
        self.query_embed = nn.Parameter(torch.randn(18, 512)) # self.class_embed
        
        #### Fusion last
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)
        
    def forward(self, images, description):
        # batch, timestamp, dim_image, height, width
        b,t,c,h,w = images.size()
        
        ###### branch1
        
        # video embedding
        x = self.backbone(images)[0]
        x = self.linear1(x)
        x = self.temperal_pool(x.transpose(1,2)).transpose(1,2)
        x = self.pos_encod(x)
        video_embedding = self.video_embedding(x.mean(dim= 1, keepdim=True))[0]
        
        # text embedding
        inputs = self.text_tokenizer(description, return_tensors = 'pt', padding = True)
        text_embedding = self.text_model(**inputs).pooler_output
        
        if video_embedding.shape[-1] != self.embed_dim:
            video_embedding = nn.Linear(video_embedding.shape[-1], self.embed_dim)(video_embedding)
        if text_embedding.shape[-1] != self.embed_dim:
            text_embedding = nn.Linear(text_embedding.shape[-1], self.embed_dim)(text_embedding)

        video_embed = video_embedding.unsqueeze(1)
        text_embed = text_embedding.unsqueeze(1)
        
        if self.fusion == 'self_attn':    
            video_self, _ = self.video_self_attn(video_embed, video_embed, video_embed)
            text_self, _ = self.text_self_attn(text_embed, text_embed, text_embed)
            fused_embedding = torch.cat([video_self.squeeze(1), text_self.squeeze(1)], dim=-1)
        elif self.fusion == 'cross_attn':
            cross_output, _ = self.cross_attn(query=text_embed, key=video_embed, value=video_embed)
            fused_embedding = cross_output.squeeze(1)
        elif self.fusion == 'transformer':
            combined = torch.cat([self.cls_token.repeat(video_embed.size(0),1,1), video_embed, text_embed], dim=1)
            # combined.shape = [batch, seq_len=3, embed_dim]

            fused_embedding = self.transformer_encoder(combined)
            
        ####### branch 2
        video_features = self.image_model(images.reshape(b * t, c, h, w))[1].reshape(b, t, -1).mean(dim=1, keepdim=True)
        query_embed = self.linear2(
            torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), 
                          video_features.repeat(1, self.num_classes, 1)), 
                        dim =2))
        
        ###### Fusion
        hs = self.transformer(fused_embedding, query_embed)
        out = self.group_linear(hs)
        
        return out

# Test case Test case Test case Test case Test case
# import torch

# # Số lượng batch và số frame
# batch_size = 1
# num_frames = 30
# channels = 3
# height, width = 224, 224

# # Dữ liệu giả lập
# dummy_images = torch.randn(batch_size, num_frames, channels, height, width)
# dummy_texts = "A man is playing football"

# # Khởi tạo mô hình
# model = TimeSformerMutimodal()

# # Kiểm tra mô hình
# output = model(dummy_images, dummy_texts)
# print("✅ Model chạy thành công! Output shape:", output.shape)

