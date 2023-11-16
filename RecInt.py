import os
import random
import time
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from transformers import LlamaForCausalLM, LlamaTokenizer

from recommender.A_SASRec_final_bce_llm import SASRec, GRU

from data_utils import *

from evaluator import *

parser = argparse.ArgumentParser(description='RecInt')
parser.add_argument("--cuda", type=str, default= "2")
parser.add_argument("--epochs", type=int, default= 100)
parser.add_argument("--lr", type=float, default= 0.001)
parser.add_argument("--weight_decay", type=float, default= 1e-6)
parser.add_argument("--rec_size", type=int, default= 64)
parser.add_argument("--recommender", type=str, default= "SASRec")
parser.add_argument("--rec_type", type=str, default= "h_all")
parser.add_argument("--LLM", type=str, default= "LLAMA1")
parser.add_argument("--data", type=str, default= "ml100k")
parser.add_argument("--save_flag", type=bool, default=True)
# parser.add_argument("--save_folder", type=str, default = "./country_result_new_test/")

args = parser.parse_args()

class RecInt(nn.Module):
    def __init__(self, 
                 recommender='SASRec',
                 rec_size=64,
                 rec_type='h_all',
                 llama_model="/data/yangzy/LLAMAHF/7B/",
                 max_txt_len=32,
                 prompt_template="{}",
                 data='ml100k',
                 end_sym='\n',
                 ):
        super().__init__()

        self.data = data
        self.rec_size = rec_size
        self.recommender = recommender
        self.rec_type = rec_type


        print('Loading Rec Model')
        if self.data == 'ml100k':
            self.rec_model = torch.load('./RecModel/{}_{}_{}.pt'.format(self.recommender, self.data, self.rec_type))
        else:
            self.rec_model = torch.load('./RecModel/SASRec_{}_h_all.pt'.format(self.data))
        self.rec_model.eval()

        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False

        print('Loding Rec model Done')

        if self.data == 'ml100k' or self.data == 'game' or self.data == 'game2' or self.data == 'steam':
            prompt_path="./prompt/{}/alignment_seq.txt".format(self.data)
            prompt_path_test="./prompt/{}/alignment_seq_test.txt".format(self.data)
        else:
            raise ValueError("no dataset: {}".format(self.data))

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<SeqHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        if prompt_path_test:
            with open(prompt_path_test, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<SeqHere>" in raw_prompt]
            self.prompt_list_test = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} test prompts'.format(len(self.prompt_list_test)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list_test)))
        else:
            self.prompt_list = []

        print('Loading LLAMA')

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = "$$"
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['<Seq>', '</Seq>']})
        # self.llama_tokenizer.mol_token_id = self.llama_tokenizer("<mol>", add_special_tokens=False).input_ids[0]
        
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        print('Loading LLAMA Done')

        self.conv = nn.Conv2d(1, 1, (10, 1))

        self.llama_proj = nn.Linear(
            self.rec_size, self.llama_model.config.hidden_size
        )

        self.act_f = nn.ReLU()
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def encode_seq(self, seq, len_seq):

        device = seq.device

        seq_emb_all = self.rec_model.cacul_h_all(seq, len_seq)

        seq_emb_conv = self.conv(seq_emb_all.view(seq_emb_all.shape[0], 1, 10, self.rec_size))

        seq_emb = seq_emb_conv.view(seq_emb_all.shape[0], 1, self.rec_size)

        # print(seq_emb.shape)
        seq_input_llama = self.llama_proj(seq_emb)
        seq_atts_llama = torch.ones(seq_input_llama.size()[:-1], dtype=torch.long).to(seq.device)

        # seq_input_llama = self.act_f(seq_input_llama)
        return seq_input_llama, seq_atts_llama
    
    def encode_seq_gru(self, seq, len_seq):

        device = seq.device

        seq_emb_all = self.rec_model.cacul_h_all(seq, len_seq)

        seq_emb_conv = self.conv(seq_emb_all.view(seq_emb_all.shape[0], 1, 10, self.rec_size))

        seq_emb = seq_emb_conv.view(seq_emb_all.shape[0], 1, self.rec_size)

        # print(seq_emb.shape)
        seq_input_llama = self.llama_proj(seq_emb)
        seq_atts_llama = torch.ones(seq_input_llama.size()[:-1], dtype=torch.long).to(seq.device)

        # seq_input_llama = self.act_f(seq_input_llama)
        return seq_input_llama, seq_atts_llama
    
    
    def get_context_emb(self, prompt, seq_list):
        device = seq_list[0].device

        prompt_segs = prompt.split('<SeqHere>')

        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], seq_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)

        return mixed_embs


    def prompt_wrap(self, seq_embeds, atts_seq, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(seq_embeds)

            for each_seq_embed, each_prompt in zip(seq_embeds, prompts):
                p_before, p_after = each_prompt.split('<SeqHere>')

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(seq_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(seq_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_seq_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=seq_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=seq_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def embed_tokens(self, token_ids):

        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds


    def forward(self, samples):
        seq = samples["seq"]
        len_seq = samples['len_seq']
        len_seq_list = samples['len_seq_list']
        if self.recommender == 'SASRec':
            seq_embeds, atts_seq = self.encode_seq(seq, len_seq)
        elif self.recommender == 'GRU':
            seq_embeds, atts_seq = self.encode_seq_gru(seq, len_seq_list)

        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        seq_embeds, atts_seq = self.prompt_wrap(seq_embeds, atts_seq, instruction)

        self.llama_tokenizer.padding_side = "right"

        movie_seq_list = [t.split('::') for t in sample["movie_seq"]]
        if self.data == 'ml100k':
            movie_seq_str = [', '.join(t) for t in movie_seq_list]
            movie_seq_str = ['This user has watched ' + t + ' in the previous.' for t in movie_seq_str]
        elif self.data == 'game' or self.data == 'game2' or self.data == 'steam':
            movie_seq_str = [', '.join(t) for t in movie_seq_list]
            movie_seq_str = ['This user has played ' + t + ' in the previous.' for t in movie_seq_str]
        else:
            raise ValueError("no dataset: {}".format(self.data))

        text = [t + self.end_sym for t in movie_seq_str]

        # print(text)

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(seq.device)

        batch_size = seq_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_seq[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(seq_embeds, atts_seq, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                    dtype=torch.long).to(seq.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target  # plus 1 for bos

        # with self.maybe_autocast():
        #     outputs = self.llama_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         return_dict=True,
        #         labels=targets,
        #     )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return loss
    

    def generate_output(self, samples):
        seq = samples["seq"]
        len_seq = samples['len_seq']
        len_seq_list = samples['len_seq_list']
        if self.recommender == 'SASRec':
            seq_embeds, atts_seq = self.encode_seq(seq, len_seq)
        elif self.recommender == 'GRU':
            seq_embeds, atts_seq = self.encode_seq_gru(seq, len_seq_list)

        if self.prompt_list_test:
            instruction = random.choice(self.prompt_list_test)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        instruction_tokens = self.llama_tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(seq_embeds.device)

        seq_embeds, atts_seq = self.prompt_wrap(seq_embeds, atts_seq, instruction)

        batch_size = seq_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=instruction_tokens.input_ids.dtype,
                        device=instruction_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_seq[:, :1]
        # atts_bos = torch.ones([batch_size, 1], dtype=torch.int, device=item_embeds.device),


        inputs_embeds = torch.cat([bos_embeds, seq_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_seq], dim=1)

        generate_ids = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=100
        )      

        response = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  

        return response
        

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.data == 'ml100k':
        training_samples = TrainingData(data4frame(), device=device)
        train_dataloader = DataLoader(training_samples, batch_size=16, shuffle=True)

        test_samples = TrainingData(data4frame_test(), device=device)
        test_dataloader = DataLoader(test_samples, batch_size=32, shuffle=False)
    elif args.data == 'steam':
        training_samples = TrainingData(data4frame_steam(), device=device)
        train_dataloader = DataLoader(training_samples, batch_size=16, shuffle=True)

        test_samples = TrainingData(data4frame_steam_test(), device=device)
        test_dataloader = DataLoader(test_samples, batch_size=32, shuffle=False)
    else:
        raise ValueError("no dataset: {}".format(args.data))
    

    print('Loading data {} done'.format(args.data))


    if args.LLM == 'LLAMA1':
        LLM_PATH = "/data/yangzy/LLAMAHF/7B/"
    if args.LLM == 'LLAMA2':
        LLM_PATH = "/data/yangzy/LLaMA2/llama-2-hf-7b/"


    RecIntModel = RecInt(recommender=args.recommender, rec_type=args.rec_type, rec_size=args.rec_size, llama_model=LLM_PATH, data=args.data)
    RecIntModel.to(device)

    
    optimizer = torch.optim.AdamW(RecIntModel.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1, total_iters=5)

    total_step=0
    for i in range(args.epochs):
        loss_total = 0
        t_begin_epoch = time.time()
        t_begin_step = time.time()
        for _, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = RecIntModel.forward(sample)
            loss.backward()
            optimizer.step()
            loss_total += loss
            total_step += 1

            if total_step % 2000 == 0:
                print('TEST STEP: {:03d}'.format(total_step))
                if args.save_flag:
                    save_dir = r'./ckpt/{}/{}/recover/'.format(args.data, args.recommender)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(RecIntModel.llama_proj, save_dir + '{}.pt'.format(total_step))

                t_begin_test = time.time()
                evaluate(RecIntModel, test_dataloader)

                print('TEST STEP: {:03d} Finished; IN EPOCH: {:03d}'.format(total_step, i))
                print('TRAIN TIME COST: ' + time.strftime("%H: %M: %S", time.gmtime(t_begin_test - t_begin_step)))
                print('TEST TIME COST: ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-t_begin_test)))
                t_begin_step = time.time()

            if total_step % 5000 == 0:
                scheduler.step()

        t_end_epoch = time.time()


        print('TEST EPOCH: {:03d}'.format(i))
        evaluate(RecIntModel, test_dataloader)
        print('TEST EPOCH: {:03d} Finished'.format(i))

        print('EPOCH: {:03d}'.format(i) + ' LOSS: {:.4f}'.format(loss_total) + " TIME: " + time.strftime(
                        "%H: %M: %S", time.gmtime(t_end_epoch-t_begin_epoch)))

    




  



