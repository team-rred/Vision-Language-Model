import torch
import torch.nn as nn
from transformers import BertConfig, AlbertConfig, AutoTokenizer, AutoModel, BertModel, AutoConfig

from models.image import ImageEncoder

class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1 
            ############ 여기서 [SEP]토큰 추가 안해도 될거같은데? class JsonlDataset(Dataset) 에서 text start token으로 [SEP] 추가 하는데 텍스트와 이미지 사이에 중복으로 [SEP]들어가는거 아닌가?
            ### --> dataset.py line 78~ 부터 해결.
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args

        if args.init_model == 'google/bert_uncased_L-4_H-512_A-8':
            bert = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
        if args.init_model == "bert-base-scratch":
            config = BertConfig.from_pretrained("bert-base-uncased")
            bert = BertModel(config)
        else:
            bert = BertModel.from_pretrained(args.init_model)
        # bert = BertModel.from_pretrained(args.init_model)#, type_vocab_size = type_vocab_size)
        self.txt_embeddings = bert.embeddings
        # old_embed = self.txt_embeddings.token_type_embeddings.weight.data
        # self.txt_embeddings.token_type_embeddings = nn.Embedding(3, self.args.embed_sz)
        # self.txt_embeddings.token_type_embeddings.weight.data[:2,:] = old_embed

        # self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings) # 여기서 [CLS] + img_embedding + [SEP] 으로 만듦
        self.img_encoder = ImageEncoder(args) #mmbt
        # self.img_encoder = fully_use_cnn()

        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, input_img):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        try:
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except StopIteration:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048

        
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        encoded_layers = self.encoder(
            encoder_input, extended_attention_mask#, output_all_encoded_layers=False
        )

        return self.pooler(encoded_layers[-1])

    def expand_token_type_embeddings(self):
        old_embed = self.txt_embeddings.token_type_embeddings.weight.data
        self.txt_embeddings.token_type_embeddings = nn.Embedding(3, self.args.embed_sz)
        self.txt_embeddings.token_type_embeddings.weight.data[:2,:] = old_embed

        self.img_embeddings = ImageBertEmbeddings(self.args, self.txt_embeddings) # 여기서 [CLS] + img_embedding + [SEP] 으로 만듦

class MultimodalBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment, img):
        x = self.enc(txt, mask, segment, img)
        return self.clf(x)
    
    def expand_token_type_embeddings(self):
        self.enc.expand_token_type_embeddings()