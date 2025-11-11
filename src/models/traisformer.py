# Copied from [CIA-Oceanix/TrAISformer](https://github.com/CIA-Oceanix/TrAISformer)

"""Models for TrAISformer.
    https://arxiv.org/abs/2109.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import pdb


import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)




class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        
        # Create default config for fallback values
        default_config = TrAISformerConfig()
        
        # Helper function to get config value with fallback to default
        def get_config(attr):
            if isinstance(config, dict):
                return config.get(attr, getattr(default_config, attr))
            else:
                return getattr(config, attr, getattr(default_config, attr))
        
        self.n_embd = config.get('n_embd', get_config('n_embd'))
        self.attn_pdrop = config.get('attn_pdrop', get_config('attn_pdrop'))
        self.resid_pdrop = config.get('resid_pdrop', get_config('resid_pdrop'))
        self.n_head = config.get('n_head', get_config('n_head'))
        self.max_seqlen = config.get('max_seqlen', get_config('max_seqlen'))
        
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(self.n_embd, self.n_embd)
        self.query = nn.Linear(self.n_embd, self.n_embd)
        self.value = nn.Linear(self.n_embd, self.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(self.attn_pdrop)
        self.resid_drop = nn.Dropout(self.resid_pdrop)
        # output projection
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(self.max_seqlen, self.max_seqlen))
                                     .view(1, 1, self.max_seqlen, self.max_seqlen))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        
        # Create default config for fallback values
        default_config = TrAISformerConfig()
        
        # Helper function to get config value with fallback to default
        def get_config(attr):
            if isinstance(config, dict):
                return config.get(attr, getattr(default_config, attr))
            else:
                return getattr(config, attr, getattr(default_config, attr))
            
        self.n_embd = config.get('n_embd', get_config('n_embd'))
        self.resid_pdrop = config.get('resid_pdrop', get_config('resid_pdrop'))
            
        super().__init__()
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(self.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model = None):
        super().__init__()
        
        # Create default config for fallback values
        default_config = TrAISformerConfig()
        
        # Helper function to get config value with fallback to default
        def get_config(attr):
            if isinstance(config, dict):
                return config.get(attr, getattr(default_config, attr))
            else:
                return getattr(config, attr, getattr(default_config, attr))

        self.lat_size = config.get('lat_size', get_config('lat_size'))
        self.lon_size = config.get('lon_size', get_config('lon_size'))
        self.sog_size = config.get('sog_size', get_config('sog_size'))
        self.cog_size = config.get('cog_size', get_config('cog_size'))
        self.full_size = config.get('full_size', get_config('full_size'))
        self.n_lat_embd = config.get('n_lat_embd', get_config('n_lat_embd'))
        self.n_lon_embd = config.get('n_lon_embd', get_config('n_lon_embd'))
        self.n_sog_embd = config.get('n_sog_embd', get_config('n_sog_embd'))
        self.n_cog_embd = config.get('n_cog_embd', get_config('n_cog_embd'))
        self.register_buffer(
            "att_sizes", 
            torch.tensor([self.lat_size, self.lon_size, self.sog_size, self.cog_size]))
        self.register_buffer(
            "emb_sizes", 
            torch.tensor([self.n_lat_embd, self.n_lon_embd, self.n_sog_embd, self.n_cog_embd]))
        
        if config.get("partition_mode", False):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model
        
        if config.get("blur", False):
            self.blur = config["blur"]
            self.blur_learnable = config["blur_learnable"]
            self.blur_loss_w = config["blur_loss_w"]
            self.blur_n = config["blur_n"]
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding = 1, padding_mode = 'replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1/3)
            else:
                self.blur_module = None
        else:
            self.blur = False
                
        
        if config.get("lat_min", False): # the ROI is provided.
            self.lat_min = config.get("lat_min", None)
            self.lat_max = config.get("lat_max", None)
            self.lon_min = config.get("lon_min", None)
            self.lon_max = config.get("lon_max", None)
            self.lat_range = self.lat_max-self.lat_min
            self.lon_range = self.lon_max-self.lon_min
            self.sog_range = 30.
            
        if config.get("mode", False): # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to 
            # calculate the next positions.
            self.mode = config["mode"]
        else:
            self.mode = "pos"
    

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, self.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, self.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, self.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, self.n_cog_embd)
            
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.get("max_seqlen", get_config("max_seqlen")), config.get("n_embd", get_config("n_embd"))))
        self.drop = nn.Dropout(config.get("embd_pdrop", get_config("embd_pdrop")))
        
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.get("n_layer", get_config("n_layer")))])
        
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.get("n_embd", get_config("n_embd")))
        if self.mode in ("mlp_pos","mlp"):
            self.head = nn.Linear(config.get("n_embd", get_config("n_embd")),
                                  config.get("n_embd", get_config("n_embd")),
                                  bias=False)
        else:
            self.head = nn.Linear(config.get("n_embd", get_config("n_embd")),
                                  self.full_size, bias=False) # Classification head
            
        self.max_seqlen = config.get("max_seqlen", get_config("max_seqlen"))
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
                                                    
        # Create default config for fallback values
        default_config = TrAISformerConfig()
        
        # Helper function to get config value with fallback to default
        def get_config(attr):
            if isinstance(train_config, dict):
                return train_config.get(attr, getattr(default_config, attr))
            else:
                return getattr(train_config, attr, getattr(default_config, attr))

        weight_decay = train_config.get("weight_decay", get_config("weight_decay"))
        learning_rate = train_config.get("learning_rate", get_config("learning_rate"))
        betas = train_config.get("betas", get_config("betas"))
        
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
   
    
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.
        
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            model: currenly only supports "uniform".
        
        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x*self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):
            
            idxs = (x*self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
#             pdb.set_trace()
            idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
            idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()                               
            return idxs, idxs_uniform
    
    
    def forward(self, x, masks = None, with_targets=False, return_loss_tuple=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if 
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:], 
                otherwise inputs = x.
        Returns: 
            logits, loss
        """
    
        if self.mode in ("mlp_pos","mlp",):
            idxs, idxs_uniform = x, x # use the real-values of x.
        else:            
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)
    
        if with_targets:
            inputs = idxs[:,:-1,:].contiguous()
            targets = idxs[:,1:,:].contiguous()
            targets_uniform = idxs_uniform[:,1:,:].contiguous()
            inputs_real = x[:,:-1,:].contiguous()
            targets_real = x[:,1:,:].contiguous()
            # Slice masks to match the input sequence length
            if masks is not None:
                masks = masks[:,:-1].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None
        
        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:,:,0]) # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:,:,1]) 
        sog_embeddings = self.sog_emb(inputs[:,:,2]) 
        cog_embeddings = self.cog_emb(inputs[:,:,3])      
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)
            
        position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
        fea = self.drop(token_embeddings + position_embeddings)
        fea = self.blocks(fea)
        fea = self.ln_f(fea) # (bs, seqlen, n_embd)
        logits = self.head(fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
        
        lat_logits, lon_logits, sog_logits, cog_logits =\
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
        
        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), 
                                       targets[:,:,2].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size), 
                                       targets[:,:,3].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size), 
                                       targets[:,:,0].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size), 
                                       targets[:,:,1].view(-1), 
                                       reduction="none").view(batchsize,seqlen)                     

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1) 
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1,1,self.lat_size)).reshape(lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1,1,self.lon_size)).reshape(lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1,1,self.sog_size)).reshape(sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1,1,self.cog_size)).reshape(cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:,:,0].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:,:,1].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:,:,2].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:,:,3].view(-1),
                                                  reduction="none").view(batchsize,seqlen)

                    lat_loss += self.blur_loss_w*blurred_lat_loss
                    lon_loss += self.blur_loss_w*blurred_lon_loss
                    sog_loss += self.blur_loss_w*blurred_sog_loss
                    cog_loss += self.blur_loss_w*blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs
                    

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)
        
            if masks is not None:
                loss = (loss*masks).sum(dim=1)/masks.sum(dim=1)
        
            loss = loss.mean()
        
        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss

class TrAISformerConfig:
    """Configuration for TrAISformer model"""
    
    # Optimizer defaults (used in configure_optimizers)
    learning_rate = 3e-4
    betas = (0.9, 0.95)  # Default Adam betas
    weight_decay = 0.1
    
    # Model architecture (these are typically required, but examples show defaults)
    n_layer = 12
    n_head = 12
    n_embd = 768
    
    # Dropout defaults
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # Sequence length
    max_seqlen = 256
    
    # Vocabulary/discretization sizes (project-specific)
    lat_size = 100
    lon_size = 100
    sog_size = 30
    cog_size = 72
    
    # Embedding dimensions (project-specific)
    n_lat_embd = 192
    n_lon_embd = 192
    n_sog_embd = 192
    n_cog_embd = 192
    
    # ROI limits for Danish maritime region
    lat_min = 54.0
    lat_max = 59.0
    lon_min = 5.0
    lon_max = 17.0
    
    # Computed during __init__ if needed
    @property
    def lat_range(self):
        return self.lat_max - self.lat_min
    
    @property
    def lon_range(self):
        return self.lon_max - self.lon_min
    
    @property
    def full_size(self):
        return self.lat_size + self.lon_size + self.sog_size + self.cog_size