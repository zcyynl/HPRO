import argparse
import os
import traceback
import numpy as np
import deepspeed
from modelscope import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from deepspeed.utils import log_dist
from tqdm import tqdm
from datetime import datetime

from pairdata import BalancedBatchSampler
from dataset import data_processing, MyDataset, MyDatasetWithDataAug
from layers import add_lora, append_llama_pro_block_group
from optimizer import WarmupExponentialLR


class PairwiseRankHead(nn.Module):
    def __init__(self, hidden_size, emb_size=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, emb_size, dtype=torch.bfloat16)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(emb_size, 1, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x.squeeze(-1)


def append_logtxt(log_path, message):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{t}\t{message}\n")


def precision_recall_at_k(y_true_np, y_score_np, k):
    k = min(k, len(y_true_np))
    if k == 0:
        return 0.0, 0.0
    idx_topk = np.argsort(-y_score_np)[:k]
    tp_in_k  = y_true_np[idx_topk].sum()
    precision_k = tp_in_k / k
    recall_k    = tp_in_k / y_true_np.sum() if y_true_np.sum() > 0 else 0.0
    return precision_k, recall_k


class TaskLayer(nn.Module):
    def __init__(self, hidden_size, embedding_size=128, dropout_rate=0.5):
        super(TaskLayer, self).__init__()
        self.custom_linear_0 = nn.Linear(in_features=hidden_size, out_features=embedding_size, dtype=torch.bfloat16)
        self.custom_linear_dropout = nn.Dropout(p=dropout_rate)
        self.custom_linear_activation = nn.GELU()
        self.custom_linear_1 = nn.Linear(in_features=embedding_size, out_features=1, dtype=torch.bfloat16)

    def forward(self, output):
        output = self.custom_linear_0(output)
        output = self.custom_linear_dropout(output)
        output = self.custom_linear_activation(output)
        output_f = self.custom_linear_1(output)
        return output_f, output


class LanguageModelWithLinear(nn.Module):
    def __init__(self, pretrained_model_name, lora_r, lora_alpha, llama_pro_group_size=-1,
    k=128, dropout_rate=0.5, cl_tau=0, pissa=False, pairwise_margin=0.0, use_hpro=False,
    hpro_margin_global=1.0, hpro_margin_key=0.5, hpro_margin_soft=0.1):
        super(LanguageModelWithLinear, self).__init__()

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            use_cache=False
        )
        log_dist(message=f"{self.pretrained_model}", ranks=[0])

        log_dist(message=f"*******LanguageModelWithLinear llama_pro group size:{llama_pro_group_size}")
        if llama_pro_group_size > 0:
            append_llama_pro_block_group(self.pretrained_model, llama_pro_group_size)
        else:
            if lora_r > 0:
                add_lora(self.pretrained_model, lora_r, lora_alpha, pissa)

        self.custom_linear = TaskLayer(self.pretrained_model.config.hidden_size, k, dropout_rate)
        self.pairwise_margin = pairwise_margin
        self.pairwise_head = PairwiseRankHead(self.pretrained_model.config.hidden_size, k, dropout_rate)

        # HPRO settings
        self.use_hpro = use_hpro
        self.hpro_margin_global = hpro_margin_global
        self.hpro_margin_key = hpro_margin_key
        self.hpro_margin_soft = hpro_margin_soft

        if use_hpro:
            log_dist(message=f"*******HPRO enabled with margins: global={hpro_margin_global}, key={hpro_margin_key}, soft={hpro_margin_soft}", ranks=[0])

        self.cl_tau = cl_tau
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        log_dist(message="*******LanguageModelWithLinear COMPLETE", ranks=[0])


    def pairwise_loss_fn(self, scores, labels, margin=0.0):
        """Standard pairwise loss (BPR or Hinge)"""
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0., device=scores.device, dtype=scores.dtype)

        s_pos = scores[pos_mask]
        s_neg = scores[neg_mask]
        diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)

        if margin == 0.0:
            loss = -torch.log(torch.sigmoid(diff)).mean()
        else:
            loss = torch.relu(margin - diff).mean()
        return loss

    def hpro_loss_fn(self, scores, labels, funnel_stages=None):
        """
        HPRO: Hierarchical Preference Ranking Optimization

        Constructs preference pairs based on funnel hierarchy:
        - Global Dominance: Lock-in (label=1) vs Defeat (label=0 with no engagement)
        - Key Action: Test Drive vs No Drive
        - Soft Signal: Long Call vs Short Call

        Args:
            scores: (B,) pairwise ranking scores
            labels: (B,) binary labels (0/1)
            funnel_stages: (B,) funnel stage indicators (optional)
                          If None, falls back to simple label-based preference
        """
        if funnel_stages is None:
            # Fallback to standard pairwise loss if no funnel info
            return self.pairwise_loss_fn(scores, labels, margin=self.hpro_margin_global)

        device = scores.device
        total_loss = torch.tensor(0., device=device, dtype=scores.dtype)
        n_pairs = 0

        # Global Dominance: Lock-in (stage=3) vs Defeat (stage=0)
        lock_mask = (funnel_stages == 3)
        defeat_mask = (funnel_stages == 0)
        if lock_mask.sum() > 0 and defeat_mask.sum() > 0:
            s_lock = scores[lock_mask]
            s_defeat = scores[defeat_mask]
            diff = s_lock.unsqueeze(1) - s_defeat.unsqueeze(0)
            loss_global = -torch.log(torch.sigmoid(diff - self.hpro_margin_global)).mean()
            total_loss += loss_global
            n_pairs += 1

        # Key Action: Test Drive (stage=2) vs No Drive (stage=1)
        drive_mask = (funnel_stages == 2)
        no_drive_mask = (funnel_stages == 1)
        if drive_mask.sum() > 0 and no_drive_mask.sum() > 0:
            s_drive = scores[drive_mask]
            s_no_drive = scores[no_drive_mask]
            diff = s_drive.unsqueeze(1) - s_no_drive.unsqueeze(0)
            loss_key = -torch.log(torch.sigmoid(diff - self.hpro_margin_key)).mean()
            total_loss += loss_key
            n_pairs += 1

        # Soft Signal: Long Call (stage=1+) vs Short Call (stage=1)
        # This is a simplification; in practice you'd use call duration metadata
        long_call_mask = (funnel_stages >= 1) & (labels == 1)
        short_call_mask = (funnel_stages >= 1) & (labels == 0)
        if long_call_mask.sum() > 0 and short_call_mask.sum() > 0:
            s_long = scores[long_call_mask]
            s_short = scores[short_call_mask]
            diff = s_long.unsqueeze(1) - s_short.unsqueeze(0)
            loss_soft = -torch.log(torch.sigmoid(diff - self.hpro_margin_soft)).mean()
            total_loss += loss_soft
            n_pairs += 1

        if n_pairs > 0:
            return total_loss / n_pairs
        else:
            return torch.tensor(0., device=device, dtype=scores.dtype)

    def cl_loss_fn(self, x, b, tau=1):
        M = x @ x.T
        x_n = torch.norm(x, dim=-1, keepdim=True)
        V = x_n @ x_n.T
        M = M[:b, b:]
        V = V[:b, b:]
        M = (M / V / tau).exp()
        M = M / M.sum(-1, keepdim=True)
        return -(torch.diagonal(M.log())).sum()

    def forward(self, input_ids, attention_mask, bce_target, ce_target, funnel_stages=None):
        b = input_ids.shape[0]
        if self.cl_tau > 0 and b > 1:
            input_ids = input_ids.repeat(2, 1)
            attention_mask = attention_mask.repeat(2, 1)
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        x, x_hidden = self.custom_linear(last_hidden_state)

        bce = x[:b].squeeze(-1)
        pair_scores = self.pairwise_head(last_hidden_state)
        ce = outputs.logits[:b, -1, :]

        cls = bce
        logits = ce

        bce_loss = self.bce_loss_fn(bce, bce_target)
        ce_loss = self.ce_loss_fn(ce, ce_target)
        loss = 2*bce_loss + 0.5*ce_loss

        # Choose between HPRO and standard pairwise loss
        if self.use_hpro and funnel_stages is not None:
            pw_loss = self.hpro_loss_fn(pair_scores, bce_target, funnel_stages)
        else:
            pw_loss = self.pairwise_loss_fn(pair_scores, bce_target, margin=self.pairwise_margin)

        loss += pw_loss

        if self.cl_tau > 0 and b > 1:
            cl_loss = self.cl_loss_fn(x_hidden, b, self.cl_tau)
            loss += cl_loss

        return loss, cls, logits, bce_loss, ce_loss, pw_loss


def eval(args, pretrained_model_name):
    ii = args.ii
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    llama_pro_group_size = args.llama_pro_group_size
    k = args.k
    dropout_rate = args.dropout_rate
    max_len = args.max_len

    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    new_model = LanguageModelWithLinear(
        pretrained_model_name, lora_r, lora_alpha, llama_pro_group_size, k, dropout_rate,
        args.cl_tau, args.pissa,
        pairwise_margin=args.pairwise_margin if args.use_pairwise else 0.0,
        use_hpro=args.use_hpro,
        hpro_margin_global=args.hpro_margin_global,
        hpro_margin_key=args.hpro_margin_key,
        hpro_margin_soft=args.hpro_margin_soft
    )
    if rank == 0:
        print(new_model)
    log_dist(message="HERE ---- -3", ranks=[0])
    train_data, test_data = data_processing(world_size=world_size, path=args.data_file)
    log_dist(message="HERE ---- -2", ranks=[0])
    dataset1 = MyDataset(train_data, tokenizer, max_len)
    log_dist(message="HERE ---- -1", ranks=[0])

    parameters = filter(lambda p: p.requires_grad, new_model.parameters())
    params_g1 = list()
    params_g2 = list()
    for param in new_model.named_parameters():
        if param[1].requires_grad:
            if param[0].startswith("custom_linear"):
                params_g1.append(param[1])
            else:
                if args.llama_pro_group_size <= 0:
                    if param[0].find("lora") == -1:
                        param[1].requires_grad = False
                    else:
                        params_g2.append(param[1])
                else:
                    if param[0].find('mlp.down_proj') >=0 or param[0].find('self_attn.o_proj') >= 0:
                        params_g1.append(param[1])
                    else:
                        params_g2.append(param[1])

    log_dist(message="HERE ---- 0", ranks=[0])
    lr2 = args.lr / 20.0
    optimizer_ = torch.optim.AdamW([
        {"params": params_g1, 'lr': args.lr, "name": "task"},
        {"params": params_g2, 'lr': lr2, "name": "llm"}
    ], lr=args.lr)

    log_dist(message=f"HERE ---- 1{args.warmup_step_num}", ranks=[0])

    lr_scheduler_ = WarmupExponentialLR(optimizer=optimizer_, warmup_step=args.warmup_step_num, gamma=0.999)
    log_dist(message="HERE ---- 2", ranks=[0])
    log_dist(message="begin create deepspeed model", ranks=[0])
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
            args=args, model=new_model, optimizer=optimizer_, lr_scheduler=lr_scheduler_,
            model_parameters=parameters, training_data=dataset1,
        )
    model_engine.load_checkpoint(load_dir=args.out_dir, tag=f"tag-ff-{ii}")
    model_engine.eval()
    evaluate(test_data=test_data, model_engine=model_engine, tokenizer=tokenizer,
             max_len=max_len, ii=ii)


def train(args, pretrained_model_name):
    threshold = args.pr_threshold

    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    logtxt_path = os.path.join(args.out_dir, "train_log.txt")
    log_dir = os.path.dirname(logtxt_path)
    os.makedirs(log_dir, exist_ok=True)
    if rank==0 and not os.path.exists(logtxt_path):
        with open(logtxt_path, 'w', encoding='utf-8') as f:
            f.write("time\tepoch\tstep\tloss\ttrain_auc\tprecision\trecall\tf1\tpos_num\tneg_num\n")

    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    llama_pro_group_size = args.llama_pro_group_size
    k = args.k
    dropout_rate = args.dropout_rate
    max_len = args.max_len

    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    new_model = LanguageModelWithLinear(
        pretrained_model_name, lora_r, lora_alpha, llama_pro_group_size, k, dropout_rate,
        args.cl_tau, args.pissa,
        pairwise_margin=args.pairwise_margin if args.use_pairwise else 0.0,
        use_hpro=args.use_hpro,
        hpro_margin_global=args.hpro_margin_global,
        hpro_margin_key=args.hpro_margin_key,
        hpro_margin_soft=args.hpro_margin_soft
    )
    if rank == 0:
        print(new_model)
    log_dist(message="HERE ---- -3", ranks=[0])
    train_data, test_data = data_processing(world_size=world_size, path=args.data_file)
    log_dist(message="HERE ---- -2", ranks=[0])
    dataset1 = MyDatasetWithDataAug(train_data, tokenizer, max_len, args.data_aug_n)
    log_dist(message="HERE ---- -1", ranks=[0])

    parameters = filter(lambda p: p.requires_grad, new_model.parameters())
    params_g1 = list()
    params_g2 = list()
    for param in new_model.named_parameters():
        if param[1].requires_grad:
            if param[0].startswith("custom_linear"):
                print(f"{param[0]} in group1")
                params_g1.append(param[1])
            else:
                if args.llama_pro_group_size <= 0:
                    if param[0].find("lora") == -1:
                        param[1].requires_grad = False
                    else:
                        print(f"{param[0]} in group2")
                        params_g2.append(param[1])
                else:
                    if param[0].find('mlp.down_proj') >=0 or param[0].find('self_attn.o_proj') >= 0:
                        params_g1.append(param[1])
                        print(f"{param[0]} in group1")
                    else:
                        params_g2.append(param[1])
                        print(f"{param[0]} in group2")

    log_dist(message="HERE ---- 0", ranks=[0])
    lr2 = args.lr / 20.0
    optimizer_ = torch.optim.AdamW([
        {"params": params_g1, 'lr': args.lr, "name": "task"},
        {"params": params_g2, 'lr': lr2, "name": "llm"}
    ], lr=args.lr)

    log_dist(message=f"HERE ---- 1{args.warmup_step_num}", ranks=[0])

    lr_scheduler_ = WarmupExponentialLR(optimizer=optimizer_, warmup_step=args.warmup_step_num, gamma=0.999)
    log_dist(message="HERE ---- 2", ranks=[0])
    log_dist(message="begin create deepspeed model", ranks=[0])

    balanced_sampler = BalancedBatchSampler(dataset1, batch_size=args.batch_size)
    train_dataloader = DataLoader(dataset1, batch_sampler=balanced_sampler)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args, model=new_model, optimizer=optimizer_, lr_scheduler=lr_scheduler_,
            model_parameters=parameters, training_data=dataset1,
        )

    for param in model_engine.named_parameters():
        if param[1].requires_grad:
            log_dist(message=f"parameters:{param[0]}", ranks=[0])
    log_dist(message="end create deepspeed model", ranks=[0])

    def get_model_engine_device(model_engine):
        return f"cuda:{model_engine.local_rank}"

    rank = get_model_engine_device(model_engine=model_engine)

    y_true_tensor = torch.tensor([]).to(rank)
    y_pred_tensor = torch.tensor([]).to(rank)
    ii = 0
    log_dist(message=f"total training set size: {len(train_dataloader)}", ranks=[0])

    device = model_engine.local_rank
    y_true_tensor = torch.tensor([], device=device)
    y_pred_tensor = torch.tensor([], device=device)

    for epoch in range(args.epochs):
        log_dist(message=f"Epoch {epoch + 1}/{args.epochs}", ranks=[0])
        for batch_idx, batch_data in enumerate(train_dataloader):
            # Unpack batch data - support both old format and new format with funnel stages
            if len(batch_data) == 4:
                input_ids, attention_mask, label, label_str = batch_data
                funnel_stages = None
            elif len(batch_data) == 5:
                input_ids, attention_mask, label, label_str, funnel_stages = batch_data
                funnel_stages = funnel_stages.to(rank)
            else:
                input_ids, attention_mask, label, label_str = batch_data[:4]
                funnel_stages = None

            input_ids = input_ids.to(rank)
            attention_mask = attention_mask.to(rank)
            label = label.to(rank)
            label_str = label_str.to(rank)

            loss, cls, logits, bce_loss, ce_loss, pw_loss = model_engine(
                input_ids, attention_mask, label, label_str, funnel_stages
            )
            cls = torch.sigmoid(cls)
            y_head = cls.detach()
            y_true_tensor = torch.hstack((y_true_tensor, label))
            y_pred_tensor = torch.hstack((y_pred_tensor, y_head))

            if ii % 10 == 0:
                log_msg = (f"step={ii} "
                           f"loss={loss.item():.6f} "
                           f"bce_loss={bce_loss.item():.6f} "
                           f"ce_loss={ce_loss.item():.6f} "
                           f"pw_loss={pw_loss.item():.6f}")
                log_dist(message=log_msg, ranks=[0])
                for ele in optimizer.param_groups:
                    log_dist(message=f"{ele['name']}train_step-{ii}\t\tlast_lr-{ele['lr']}",ranks=[0])

                if model_engine.monitor.tb_monitor is not None:
                    sw = model_engine.monitor.tb_monitor.summary_writer
                    sw.add_scalar("loss/total", loss.item(), ii)
                    sw.add_scalar("loss/bce", bce_loss.item(), ii)
                    sw.add_scalar("loss/ce", ce_loss.item(), ii)
                    sw.add_scalar("loss/pw", pw_loss.item(), ii)

            y_pred_label = (y_head > threshold).to(torch.int8).detach().cpu().numpy()
            label1 = label.to(torch.int8).detach().cpu().numpy()
            p_val = precision_score(label1, y_pred_label, zero_division=0)
            r_val = recall_score(label1, y_pred_label,  zero_division=0)

            if ii % 10 == 0:
                log_dist(message=f"precision@{threshold}:{p_val:.4f}  "
                                 f"recall@{threshold}:{r_val:.4f}", ranks=[0])

            if (ii+1) % 100 == 0:
                log_dist(message=f"{y_true_tensor.detach().cpu()}", ranks=[0])
                log_dist(message=f"{y_pred_tensor.detach().cpu()}", ranks=[0])
                auc = roc_auc_score(y_true_tensor.cpu().float().numpy()[-256:],
                                    y_pred_tensor.cpu().float().numpy()[-256:])

                log_dist(message=f"\n\n\nAUC:{auc}\t\t\t\t--------Step:{ii}", ranks=[0])
                if model_engine.local_rank == 0:
                    if model_engine.monitor.tb_monitor is not None:
                        model_engine.monitor.tb_monitor.summary_writer.add_scalar("train_auc", auc, ii+1)

            model_engine.backward(loss)
            model_engine.step()

            if ii % args.ckpt_interval == 0:
                n = len(test_data) // 10
                test_data1 = test_data.sample(n, random_state=42).reset_index(drop=True)

                evaluate(test_data=test_data1, model_engine=model_engine, tokenizer=tokenizer, max_len=max_len, ii=ii)

                client_dict = {"loss": loss}

                model_engine.save_checkpoint(save_dir=args.out_dir,
                                    client_state=client_dict,
                                    tag=f"tag-ff-{ii}",
                                    save_latest=True)

            if (ii+1) % 100 == 0 and model_engine.local_rank == 0:
                y_true_np = y_true_tensor.cpu().float().numpy()
                y_pred_np = y_pred_tensor.cpu().float().numpy()

                auc = roc_auc_score(y_true_np, y_pred_np)
                y_pred_label = (y_pred_np > 0.5).astype(int)

                pre = precision_score(y_true_np, y_pred_label, zero_division=0)
                rec = recall_score(y_true_np, y_pred_label, zero_division=0)
                f1 = f1_score(y_true_np, y_pred_label, zero_division=0)

                K_LIST = [100,200,300, 500, 1000]
                topk_strs = []
                for k_val in K_LIST:
                    p_k, r_k = precision_recall_at_k(y_true_np, y_pred_np, k_val)
                    topk_strs.append(f"P@{k_val}={p_k:.4f},R@{k_val}={r_k:.4f}")
                topk_info = " | ".join(topk_strs)
                log_dist(message=f"[Top-k] {topk_info}", ranks=[0])

                pos_num = y_true_np.sum()
                neg_num = len(y_true_np) - pos_num
                txt = f"{epoch}\t{ii}\t{loss:.4f}\t{auc:.4f}\t{pre:.4f}\t{rec:.4f}\t{f1:.4f}\t{int(pos_num)}\t{int(neg_num)}\t{topk_info}"
                append_logtxt(logtxt_path, txt)

            ii += 1

    client_dict = {"loss": loss}

    model_engine.save_checkpoint(save_dir=args.out_dir,
                                client_state=client_dict,
                                tag=f"tag-ff",
                                save_latest=True)

    evaluate(test_data=test_data, model_engine=model_engine, tokenizer=tokenizer,
             max_len=max_len, ii=len(train_dataloader)*args.epochs+1,threshold=args.pr_threshold)


def evaluate(test_data, model_engine, tokenizer, max_len, ii,threshold=0.1):
    eval_logtxt_path = os.path.join(args.out_dir, "eval_log.txt")
    if model_engine.local_rank == 0 and not os.path.exists(eval_logtxt_path):
        with open(eval_logtxt_path, 'w', encoding='utf-8') as f:
            f.write("time\tstep\teval_loss\teval_auc\tprecision\trecall\tf1\tpos_num\tneg_num\n")

    log_dist(message=f"\n\n\n ----------- Evaluation -------------- \n\n\n", ranks=[0])
    rank = model_engine.local_rank
    y_true_test = torch.tensor([]).to(rank)
    y_pred_test = torch.tensor([]).to(rank)
    dataset_test = MyDataset(test_data, tokenizer, max_len)

    log_dist(message=f"Evaluation Steps 1 is {len(dataset_test)}", ranks=[0])
    dataset_sampler = DistributedSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  sampler=dataset_sampler)
    log_dist(message=f"Evaluation Steps 2 is {len(dataloader_test)}", ranks=[0])
    model_engine.eval()
    for batch_idx, batch_data in enumerate(dataloader_test):
        if len(batch_data) == 4:
            input_ids, attention_mask, label, label_str = batch_data
            funnel_stages = None
        elif len(batch_data) == 5:
            input_ids, attention_mask, label, label_str, funnel_stages = batch_data
            funnel_stages = funnel_stages.to(rank) if funnel_stages is not None else None
        else:
            input_ids, attention_mask, label, label_str = batch_data[:4]
            funnel_stages = None

        input_ids = input_ids.to(rank)
        attention_mask = attention_mask.to(rank)
        label = label.to(rank)
        label_str = label_str.to(rank)

        with torch.no_grad():
            loss, cls, logits, bce_loss, ce_loss, pw_loss = model_engine(
                input_ids, attention_mask, label, label_str, funnel_stages
            )
            cls = torch.sigmoid(cls)
            y_head = cls.detach()
            y_true_test = torch.hstack((y_true_test, label))
            y_pred_test = torch.hstack((y_pred_test, y_head))
        if batch_idx % 10 == 0:
            log_dist(message=f"eval_step-{batch_idx}---loss:{loss}", ranks=[0])

    y_true_test_all_gather = [torch.zeros_like(y_true_test) for i in range(model_engine.world_size)]
    torch.distributed.all_gather(y_true_test_all_gather, y_true_test)

    y_pred_test_all_gather = [torch.zeros_like(y_pred_test) for i in range(model_engine.world_size)]
    torch.distributed.all_gather(y_pred_test_all_gather, y_pred_test)
    y_true = torch.cat(tensors=y_true_test_all_gather, dim=0)
    y_score = torch.cat(tensors=y_pred_test_all_gather, dim=0)
    log_dist(message=f"y_true:{y_true.detach().cpu().numpy()}", ranks=[0])
    log_dist(message=f"y_score:{y_score.detach().cpu().numpy()}", ranks=[0])

    AUC = roc_auc_score(y_true=y_true.detach().cpu().numpy(),
                        y_score=y_score.detach().cpu().float().numpy())

    y_pred_label = (y_score > threshold).long()
    P = precision_score(y_true.cpu().numpy(),
                        y_pred_label.cpu().numpy(), zero_division=0)
    R = recall_score(y_true.cpu().numpy(),
                     y_pred_label.cpu().numpy(), zero_division=0)
    cm = confusion_matrix(y_true.cpu().numpy(),
                          y_pred_label.cpu().numpy())
    log_dist(message=(
        f"\n\n\nEvaluation AUC:{AUC:.6f}  "
        f"P@{threshold}:{P:.4f}  R@{threshold}:{R:.4f}\n"
        f"Confusion_matrix:\n{cm}"
    ), ranks=[0])

    if model_engine.local_rank == 0:
        if model_engine.monitor.tb_monitor is not None:
            model_engine.monitor.tb_monitor.summary_writer.add_scalar("eval_auc", AUC, ii + 1)
            sw = model_engine.monitor.tb_monitor.summary_writer
            sw.add_scalar("eval_auc",  AUC, ii+1)
            sw.add_scalar("eval_precision", P, ii+1)
            sw.add_scalar("eval_recall",    R, ii+1)

    model_engine.train()
    log_dist(message="----------------Evaluation Completed!!!----------------", ranks=[0])

    y_true_np = y_true.detach().cpu().float().numpy()
    y_score_np = y_score.detach().cpu().float().numpy()
    y_label = (y_score_np > 0.5).astype(int)
    pre = precision_score(y_true_np, y_label, zero_division=0)
    rec = recall_score(y_true_np, y_label, zero_division=0)
    f1 = f1_score(y_true_np, y_label, zero_division=0)

    K_LIST = [100, 500, 1000]
    topk_strs = []
    for k_val in K_LIST:
        p_k, r_k = precision_recall_at_k(y_true_np, y_score_np, k_val)
        topk_strs.append(f"P@{k_val}={p_k:.4f},R@{k_val}={r_k:.4f}")
    topk_info = " | ".join(topk_strs)
    log_dist(message=f"[Eval Top-k] {topk_info}", ranks=[0])

    pos_num = y_true_np.sum()
    neg_num = len(y_true_np) - pos_num
    evalloss = float(loss.detach().cpu().numpy()) if 'loss' in locals() else 0.0
    txt = f"{ii}\t{evalloss:.4f}\t{AUC:.4f}\t{pre:.4f}\t{rec:.4f}\t{f1:.4f}\t{int(pos_num)}\t{int(neg_num)}\t{topk_info}"
    append_logtxt(eval_logtxt_path, txt)


def add_argument():
    parser = argparse.ArgumentParser(description="HPRO: Hierarchical Preference Ranking Optimization")

    parser.add_argument("--out_dir", default="", type=str, help="where to save ckpt")
    parser.add_argument("--data_file", default="/lpai/data/content_and_feat_baseline_0418.snappy.parquet", type=str, help="train data file")
    parser.add_argument("--pretrained_model", default="/lpai/llm_repo/qwen/Qwen1___5-1___8B", type=str, help="base LLM")

    parser.add_argument("--batch_size", default=32, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--epochs", default=10, type=int, help="number of total epochs (default: 30)")
    parser.add_argument("--lr", default=1.0E-4, type=float, help="learning rate")
    parser.add_argument("--ckpt_interval", default=10240, type=int, help="save checkpoint at a given interval")
    parser.add_argument("--warmup_step_num", default=4000, type=int, help="warmup step num")
    parser.add_argument("--local_rank", default=-1, type=int, help="local rank passed from distributed launcher ")
    parser.add_argument("--log-interval", default=20, type=int, help="output logging information at a given interval")

    parser.add_argument("--lora_r", default=16, type=int, help="lora rank")
    parser.add_argument("--lora_alpha", default=1.0, type=float, help="lora_alpha")
    parser.add_argument('--pissa', default=False, action='store_true',help="use PiSSA initialization for LoRA")
    parser.add_argument("--max_len", default=3600, type=int, help="max_len")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="dropout rate")
    parser.add_argument("--k", default=128, type=int, help="hidden size of task layers")
    parser.add_argument("--cl_tau", default=0, type=float, help="contrastive learning loss tau")
    parser.add_argument('--llama_pro_group_size', default=0, type=int, help="llama pro group size, 0 means no llama")
    parser.add_argument('--do_eval', default=False, action='store_true',help="only eval")
    parser.add_argument("--pr_threshold", default=0.5, type=float, help="threshold for precision / recall")
    parser.add_argument("--use_pairwise", action="store_true", help="use Pairwise ranking loss")
    parser.add_argument("--pairwise_margin", type=float, default=0.0, help=">0 for hinge loss, =0 for BPR-logistic")
    parser.add_argument('--ii', default=0, type=int,help="tag_step")
    parser.add_argument('--data_aug_n', default=0, type=int, help="data augmentation degree")

    # HPRO-specific arguments
    parser.add_argument('--use_hpro', action='store_true', help="Enable HPRO (Hierarchical Preference Ranking Optimization)")
    parser.add_argument("--hpro_margin_global", type=float, default=1.0, help="HPRO margin for global dominance (lock vs defeat)")
    parser.add_argument("--hpro_margin_key", type=float, default=0.5, help="HPRO margin for key action (drive vs no-drive)")
    parser.add_argument("--hpro_margin_soft", type=float, default=0.1, help="HPRO margin for soft signal (long call vs short call)")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = add_argument()
    if args.do_eval:
        eval(args, pretrained_model_name=args.pretrained_model)
        exit(0)
    train(args, pretrained_model_name=args.pretrained_model)
