from modules.libraries import *
from modules.cnn_mae import *
from modules.utils import *
import torchmetrics

class VideoClassificationModel(pl.LightningModule): # Model 1
    """
            Arguments:
                d_model: sequence or embedding length, default 512
                nhead: number of head for multi-head attention, default 2, 
                num_layers: number of encoder layer in transfromer encoder, default 2,
                time_stamp: total number of time stamp in each video, default 20,
                dropout: dropout ratio of mae embedding with class token + positional embeddings, default 0.1,
                num_class: number of classes/labels, default 2
    """
    def __init__(self, 
                 d_model:int = 512,
                 nhead:int = 2, 
                 num_layers:int = 2,
                 time_stamp:int = 20,
                 dropout:float = 0.1,
                 num_class:int = 2, 
                 optimizer:str = 'ADAM',
                 learning_rate:float = 0.001,
                 factor:float = 0.1,
                 patience:int = 10, 
                 required_model = None,
                 model_path = None,
                 frozen_backbone = True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        # self.d_model, self.nhead, self.num_layers = d_model, nhead, num_layers
        self.optimizer, self.learning_rate, self.factor, self.patience = optimizer, learning_rate, factor, patience
        self.frozen_backbone = frozen_backbone
        self.is_log_image = True
        # initiate the vit model, set the classifier as identity and frozen all parameters
        # only inititiate model as classfication model for now as output [batch, sequence, last_hidden_state], later model will  loaded as [batch, sequence, sequence, sequence_length]
        self.raw_encoder_model = get_pretrained_encdec(
                model_name = required_model,
                model_path = os.path.join(model_path, 'raw_cnn_mae'),  
            ).encoder_module
        self.raw_decoder_model = get_pretrained_encdec(
                model_name = required_model,
                model_path = os.path.join(model_path, 'raw_cnn_mae'),  
            ).decoder_module
        

        self.landmark_encoder_model = get_pretrained_encdec(
                model_name = required_model,
                model_path = os.path.join(model_path, 'landmark_cnn_mae'),  
            ).encoder_module
        self.landmark_decoder_model = get_pretrained_encdec(
                model_name = required_model,
                model_path = os.path.join(model_path, 'landmark_cnn_mae'),  
            ).decoder_module

        if self.frozen_backbone:
            self.raw_encoder_model = self.freeze_parameters(self.raw_encoder_model)
            self.landmark_encoder_model = self.freeze_parameters(self.landmark_encoder_model)
        self.raw_decoder_model = self.freeze_parameters(self.raw_decoder_model)
        self.landmark_decoder_model = self.freeze_parameters(self.landmark_decoder_model)

        self.cnn_layer = nn.Conv2d(in_channels = 2*512, out_channels=6, kernel_size= 3, stride = 1,  padding=1) #<------ hard coded

        # learnable class token
        self.class_token = nn.Parameter(torch.rand(1, 1, d_model), requires_grad = True)        #<------ d_model is hard coded
                                              
        
        # create positional embedding
        self.positional_embedding = nn.Parameter(torch.rand(1, time_stamp+1, d_model))         #<------ time_stamp+1, d_model are hard coded

        # create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)

        # Tranformer encoder model to find temporal relation and proceed to MLP
        self.encoder = self.Transformer_encoder( d_model, nhead, num_layers ) 

        #create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape= d_model),
            nn.Linear(in_features= d_model,
                            out_features= num_class)
        )

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # metrics
        self.accuracy = torchmetrics.Accuracy(task = "binary" if num_class==2 else 'multiclass', num_classes = num_class)
        self.f1_score = torchmetrics.F1Score(task = "binary" if num_class==2 else 'multiclass', num_classes = num_class)
        self.precision = torchmetrics.Precision(task = "binary" if num_class==2 else 'multiclass', num_classes = num_class)
    
    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def Transformer_encoder(self, d_model, nhead, num_layers):
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, batch_first= True, dim_feedforward= d_model*2, dropout=0.5)
        return nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
    
    def CLIPPORT(self, raw_input, landmark_input):
        """
        raw_input: shape [t, c, h, w]
        landmark_input: shape [t, c, h, w]
        here the t will work as batch
        """
        raw_input_, landmark_input_ = raw_input, landmark_input
        for idx in range(len(self.raw_encoder_model.encoder)):
            raw_output = self.raw_encoder_model.encoder[idx](raw_input_)
            landmark_output = self.landmark_encoder_model.encoder[idx](landmark_input_)
            raw_input_, landmark_input_ = raw_output+landmark_output, landmark_output
        return raw_output.unsqueeze(0), landmark_output.unsqueeze(0)
    

    def _log_image(self, original_img: torch.Tensor, 
                   original_recon_img: torch.Tensor, 
                   landmark_img: torch.Tensor, 
                   landmark_recon_img: torch.Tensor) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        std = torch.tensor([0.229, 0.224, 0.225]).reshape( 3, 1, 1).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape( 3, 1, 1).to(device)

        tb_logger = self.logger.experiment
        cat_original_img = original_img*std+mean
        cat_original_recon_img = original_recon_img*std+mean
        cat_landmark_img = landmark_img*std+mean
        cat_landmark_recon_img = landmark_recon_img*std+mean
        # cat_original_img = original_img[-1]*255.0
        # cat_masked_img = masked_img[-1].to(device)*255.0
        # cat_predicted_img = predicted_img[-1]*255.0
        cat_log_img = torch.cat([cat_original_img.to(device), 
                                 cat_original_recon_img.to(device), 
                                 cat_landmark_img.to(device),
                                 cat_landmark_recon_img.to(device)], dim=-1)

        tb_logger.add_image(f'Output of the: {self.current_epoch}',
                            cat_log_img, self.current_epoch, dataformats='CHW')



    def forward(self, x):
        # get some dimension from the x
        b, t, c, h, w = x.shape
        t1 = int(t/2)

        # collect the mae output per batch and then concatenate
        if self.frozen_backbone:
            with torch.no_grad():
                for idx in range(b):
                    if idx == 0:
                        raw_spatial_features_, landmark_spatial_features_ = self.CLIPPORT(x[idx, :t1, :, :, :], x[idx, t1:, :, :, :])
                    else:
                        raw_output, landmark_output = self.CLIPPORT(x[idx, :t1, :, :, :], x[idx, t1:, :, :, :])
                        raw_spatial_features_ = torch.cat( (raw_spatial_features_, raw_output), 0)
                        landmark_spatial_features_ = torch.cat( (landmark_spatial_features_, landmark_output), 0)
                del raw_output, landmark_output
        else:
            for idx in range(b):
                if idx == 0:
                    raw_spatial_features_, landmark_spatial_features_ = self.CLIPPORT(x[idx, :t1, :, :, :], x[idx, t1:, :, :, :])
                else:
                    raw_output, landmark_output = self.CLIPPORT(x[idx, :t1, :, :, :], x[idx, t1:, :, :, :])
                    raw_spatial_features_ = torch.cat( (raw_spatial_features_, raw_output), 0)
                    landmark_spatial_features_ = torch.cat( (landmark_spatial_features_, landmark_output), 0)
            # del raw_output, landmark_output

        if self.is_log_image:
            with torch.no_grad():
                raw_recon = self.raw_decoder_model(raw_spatial_features_[-1, int(t1/2), :, :, :].unsqueeze(0))
                landmark_recon = self.landmark_decoder_model(landmark_spatial_features_[-1, int(t1/2), :, :, :].unsqueeze(0))
            self._log_image(x[-1, int(t1/2), :, :, :], raw_recon[-1], x[-1, t1+int(t1/2), :, :, :], landmark_recon[-1])
            del raw_recon, landmark_recon
            self.is_log_image = False

        spatial_features_ = torch.cat((raw_spatial_features_, landmark_spatial_features_), dim = 2)
        # print(raw_spatial_features_.shape, landmark_spatial_features_.shape, spatial_features_.shape)
        del raw_spatial_features_, landmark_spatial_features_
        # cnn layer for dimention reduction
        for idx in range(b):
            if idx == 0:
                spatial_features = torch.unsqueeze(self.cnn_layer(spatial_features_[idx, :, :, :, :]), 0)
            else:
                spatial_features = torch.cat( (spatial_features, torch.unsqueeze(self.cnn_layer(spatial_features_[idx, :, :, :, :]), 0)), 0 )

        # reshape the last three dimension as the d_model of the transformers layer
        spatial_features = einops.rearrange(spatial_features, 'b t c h w->b t (c h w)')

        # prepend the class token to the mae output embedding
        class_token = self.class_token.expand(b, -1, -1)            #<------- [batch_size, 1, sequence_length]
        x = torch.cat((class_token, spatial_features), dim= 1)

        # add positional embedding
        x += self.positional_embedding

        # dropout on patch +positional embedding
        x = self.embedding_dropout(x)
        
        # pass embedding through transformer encoder
        x = self.encoder(x)  # output shape [batch, 1+timestamp, sequence_length]

        # # pass 0th index of x through MLP head
        x = self.mlp_head(x[:, 0])
        
        return x
    
    def on_train_epoch_end(self):
        self.is_log_image = True
        torch.cuda.empty_cache()
    
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        precision = self.precision(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 'train_precision': precision},
                      on_step = False, on_epoch = True, prog_bar=True, logger = True)
        return {'loss': loss, 'scores': scores, 'y': y}
    
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        precision = self.precision(scores, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_score': f1_score, 'val_precision': precision},
                      on_step = False, on_epoch = True, prog_bar=True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        precision = self.precision(scores, y)
        self.log_dict({'ttest_loss': loss, 'test_accuracy': accuracy, 'test_f1_score': f1_score, 'test_precision': precision},
                      on_step = False, on_epoch = True, prog_bar=True, logger = True)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y
    
    def predict_step(self, batch, batch_idx): 
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        if self.optimizer == 'ADAM':
            # return optim.Adam(self.parameters(), lr = self.learning_rate)
            optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.factor, patience=self.patience
    )
            return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
        else:
            optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.factor, patience=self.patience
    )
            return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

class VideoAbstractEmdModel(nn.Module):
    def __init__(self, input_feature_channel=16, embed_dim=512, pooling_size=4, num_heads=4, num_layers=2):
        super(VideoAbstractEmdModel, self).__init__()
        self.pooling_size = pooling_size
        self.input_feature_channel = input_feature_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 🔹 Feature Extraction (CNN + Adaptive Pooling)
        self.feature_extractor = nn.AdaptiveAvgPool2d((self.pooling_size, self.pooling_size))
        self.feature_dim = self.input_feature_channel * self.pooling_size ** 2

        # 🔹 Positional Encoding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 150, self.feature_dim))

        # 🔹 Transformer Encoder for Temporal Modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads, batch_first=True,
                                                   dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 🔹 Self-Attention Pooling
        self.pooling_query = nn.Parameter(torch.randn(1, self.feature_dim))  # Learnable query vector
        self.attn_fc = nn.Linear(self.feature_dim, self.feature_dim)  # Linear projection for attention

        # 🔹 Output Projection
        self.fc_out = nn.Linear(self.feature_dim, self.embed_dim)

    def extract_features(self, video_frames):
        """
        Extract spatial features from input frames.
        Args:
            video_frames: Tensor (batch_size, max_frames, C, H, W)
        Returns:
            Extracted feature tensor (batch_size, max_frames, feature_dim)
        """
        batch_size, num_frames, C, H, W = video_frames.shape

        # Flatten batch & frames to apply pooling
        frames = video_frames.view(batch_size * num_frames, C, H, W)
        pooled_features = self.feature_extractor(frames)  # Apply Adaptive Pooling
        pooled_features = pooled_features.reshape(batch_size, num_frames, -1)  # Reshape to (B, max_frames, feature_dim)

        return pooled_features

    def apply_transformer(self, features, frame_masks):
        """
        Applies transformer-based temporal modeling.
        Args:
            features: Feature tensor (batch_size, max_frames, feature_dim)
            frame_masks: Mask tensor (batch_size, max_frames)
        Returns:
            Transformed feature tensor (batch_size, max_frames, feature_dim)
        """
        src_key_padding_mask = frame_masks == 0  # True for padding, False for real frames
        transformed_features = self.transformer(features, src_key_padding_mask=src_key_padding_mask)

        return transformed_features

    def apply_self_attention_pooling(self, transformed_features, frame_masks):
        """
        Applies self-attention pooling over the transformed features.
        Args:
            transformed_features: Tensor (batch_size, max_frames, feature_dim)
            frame_masks: Tensor (batch_size, max_frames)
        Returns:
            Pooled embedding (batch_size, feature_dim)
        """
        batch_size = transformed_features.shape[0]

        # Expand query to match batch size
        query = self.pooling_query.expand(batch_size, -1).unsqueeze(1)  # (B, 1, feature_dim)

        # Compute attention scores
        attn_scores = torch.bmm(query, self.attn_fc(transformed_features).transpose(1, 2))  # (B, 1, max_frames)
        attn_scores = attn_scores.squeeze(1)  # (B, max_frames)
        attn_scores = attn_scores.masked_fill(frame_masks == 0, float('-inf'))  # Ignore padded frames
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize attention weights

        # Weighted sum of frame features
        pooled_embedding = torch.bmm(attn_weights.unsqueeze(1), transformed_features).squeeze(1)  # (B, feature_dim)

        return pooled_embedding

    def forward(self, video_frames, frame_masks):
        """
        Forward pass for video-based embedding generation.
        Args:
            video_frames: Tensor (batch_size, max_frames, C, H, W)
            frame_masks: Tensor (batch_size, max_frames) -> 1 for real frames, 0 for padded
        Returns:
            Fixed-size embedding (batch_size, embed_dim)
        """
        # 🔹 Step 1: Extract Spatial Features
        features = self.extract_features(video_frames)  # (B, max_frames, feature_dim)

        # 🔹 Step 2: Add Positional Encoding
        features = features + self.pos_embedding

        # 🔹 Step 3: Apply Transformer for Temporal Modeling
        transformed_features = self.apply_transformer(features, frame_masks)  # (B, max_frames, feature_dim)

        # 🔹 Step 4: Self-Attention Pooling
        pooled_embedding = self.apply_self_attention_pooling(transformed_features, frame_masks)  # (B, feature_dim)

        # 🔹 Step 5: Project to Embedding Space
        return self.fc_out(pooled_embedding)  # (B, embed_dim)
    
class Pretraining_memomry_MLP_Cosmos_V2(pl.LightningModule):
    def __init__(self, 
                 input_feature_channel=16,
                 d_model:int=2352,
                 learning_rate:float = 0.001,
                 factor:float = 0.1,
                 patience:int = 10,
                 optimizer:str = 'ADAM', 
                 all_classes:list=[], 
        ):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.learning_rate, self.factor, self.patience, self.optimizer = learning_rate, factor, patience, optimizer
        self.num_classes = len(all_classes)
        self.raw_fixed_abstract_model = VideoAbstractEmdModel(input_feature_channel = input_feature_channel,
                                                          embed_dim=self.d_model)
        self.landmark_fixed_abstract_model = VideoAbstractEmdModel(input_feature_channel = input_feature_channel,
                                                          embed_dim=self.d_model)
        
        #create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape= 2*self.d_model),
            nn.Dropout(0.5),
            nn.Linear(in_features= 2*self.d_model,
                            out_features= self.num_classes)
        )

                
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes)
        self.test_results = {
            'ground_truth' : [],
            'prediction' : [],
            'probabilities' : []
        }

    def forward(self, input, mask_frame):
        # unpack the raw and landmark embedding from the tuple
        raw_abstract_embedding, landmark_abstract_embedding = input
        # convert the bffloat16 to float16, bffloat16 is the data type of the COSMOS embedding
        # raw_abstract_embedding, landmark_abstract_embedding =raw_abstract_embedding.to(torch.float32), landmark_abstract_embedding.to(torch.float32)

        # feed the embedding to the fixed embedding generator
        raw_embedding = self.raw_fixed_abstract_model(raw_abstract_embedding, mask_frame)
        landmark_embedding = self.landmark_fixed_abstract_model(landmark_abstract_embedding, mask_frame)

        # Concatenate the embedding to get the abstract embedding
        abstract_embedding  =  torch.cat((raw_embedding, landmark_embedding), 1)    # [batch, 2*d_model]

        # pretrain_memory_mlp_out = self.pretrained_memomry_mlp(abstract_embedding)
        mlp_out = self.mlp_head(abstract_embedding)
        return mlp_out

    def training_step(self, batch, batch_idx):
        x, y, mask_frame = batch
        scores = self.forward(x, mask_frame)
        loss = self.loss_fn(scores, y)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        precision = self.precision(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 'train_precision': precision},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'scores': scores, 'y': y}

    def validation_step(self, batch, batch_idx):
        x, y, mask_frame = batch
        y = y.to(self.device)
        scores = self.forward(x, mask_frame)
        loss = self.loss_fn(scores, y)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        precision = self.precision(scores, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_score': f1_score, 'val_precision': precision},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask_frame = batch
        scores = self.forward(x, mask_frame)
        loss = self.loss_fn(scores, y)
        self.test_results['ground_truth'].extend(y.cpu().numpy().tolist())
        self.test_results['prediction'].extend(torch.argmax(scores, dim = 1).cpu().tolist())
        self.test_results['probabilities'].append(torch.nn.functional.softmax(scores, dim=1).cpu().tolist())
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        precision = self.precision(scores, y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1_score': f1_score, 'test_precision': precision},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_end(self):
        # Log the learning rate at the end of the epoch
        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        if self.optimizer == 'ADAM':
            # return optim.Adam(self.parameters(), lr = self.learning_rate)
            optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.factor, patience=self.patience
    )
            return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
        else:
            optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.factor, patience=self.patience
    )
            return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}]
        
class Cosmos_V2_Weighted_Loss(Pretraining_memomry_MLP_Cosmos_V2):
    def __init__(self, memory_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.load_memory_model(memory_model_path, *args, **kwargs)


    def load_memory_model(self, memory_model_path, *args, **kwargs):
        # # Load the model from the checkpoint
        if os.path.exists(memory_model_path):
            print(f'model loaded from {memory_model_path}')
            model = Pretraining_memomry_MLP_Cosmos_V2.load_from_checkpoint(memory_model_path)
        else:
            print(f'Checkpoint not found at {memory_model_path}, initializing from scratch.')
            model = Pretraining_memomry_MLP_Cosmos_V2(*args, **kwargs)
        return model
    
    def weighted_loss(self, scores, y):
        half_batch_size = len(y) // 2  # Ensure integer division

        # Split your batch into memory and new data portions
        memory_y = y[:half_batch_size]
        memory_scores = scores[:half_batch_size]
        
        new_y = y[half_batch_size:]
        new_scores = scores[half_batch_size:]
        # Calculate losses separately
        L_memory = self.loss_fn(memory_scores, memory_y)
        L_new = self.loss_fn(new_scores, new_y)
        loss = (L_memory**2 + L_new**2) / (L_memory + L_new + 1e-8)
        return loss

    
    def training_step(self, batch, batch_idx):
        x, y, mask_frame = batch
        scores = self.forward(x, mask_frame)
        loss = self.weighted_loss(scores, y)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        precision = self.precision(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 'train_precision': precision},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'scores': scores, 'y': y}
    

    def configure_optimizers(self):
        # Create two parameter groups: one for mlp_head, one for the rest
        mlp_head_params = list(self.mlp_head.parameters())
        other_params = [p for n, p in self.named_parameters() if "mlp_head" not in n and p.requires_grad]

        optimizer = None
        if self.optimizer == 'ADAM':
            optimizer = optim.Adam([
                {'params': mlp_head_params, 'lr': self.learning_rate},
                {'params': other_params, 'lr': self.learning_rate * 0.1}
            ])
        else:
            optimizer = optim.SGD([
                {'params': mlp_head_params, 'lr': self.learning_rate},
                {'params': other_params, 'lr': self.learning_rate * 0.1}
            ])

        # Scheduler remains the same, monitoring "val_loss"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.factor, patience=self.patience
        )

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
