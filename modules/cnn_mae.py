from modules.libraries import *
from modules.utils import *

class EncoderModule(nn.Module):
    def __init__(self, image_height:int, 
                input_channel:int, 
                 encoder_output_shape:list,
                #  channel_multiply:int = 32
                 ):
        super(EncoderModule, self).__init__()

        assert image_height % encoder_output_shape[-1] == 0, f'encoder input shape cannot be downsampled by the power of 2 to match the encoder output size {self.image_height}'
        
        self.num_encode_block = int( math.log10( int(image_height / encoder_output_shape[-1]) ) / math.log10(2) )

        assert encoder_output_shape[0] % self.num_encode_block == 0, f'encoder input shape cannot be downsampled by the power of 2 to match the encoder output size {self.image_height}'

        self.first_cnv2d_outchannel = int(encoder_output_shape[0] / (2**self.num_encode_block))

        self.input_channel = input_channel

        self.encoder = self._encoder()



    def _encoder(self):

        encoder = nn.Sequential()

        encoder.add_module(f'encoder_block1', nn.Conv2d(in_channels= self.input_channel, 
                                                                        out_channels= self.first_cnv2d_outchannel, 
                                                                        kernel_size=3, 
                                                                        stride=1, 
                                                                        padding=1, 
                                                                        bias=True))

        for idx in range(1, self.num_encode_block+1):

            if idx==1:
                encoder.add_module(f'encoder_block{idx+1}', self._encoder_block(in_channels = self.first_cnv2d_outchannel, 
                                                                            out_channels = int(self.first_cnv2d_outchannel*int(2**idx)), stride =2, dropout=0.5))
            else:
                encoder.add_module(f'encoder_block{idx+1}', self._encoder_block(in_channels = int(self.first_cnv2d_outchannel*int(2**(idx-1))), 
                                                                            out_channels = int(self.first_cnv2d_outchannel*int(2**idx)), stride =2, dropout= 0.5))
        
        return encoder
    

        

    def _encoder_block(self, in_channels = 256, out_channels = 512, stride =1, dropout = 0):
        encoder_block = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False),
        nn.Dropout2d(dropout),
        ResidualBlock(in_channels, out_channels),
        # nn.UpsamplingBilinear2d(scale_factor=2)
        )
        return encoder_block

    def forward(self, x):
        # print(x.shape)
        # x = self.decoder.decoder_block1(x)
        # print(x.shape)
        # x = self.decoder.decoder_block2(x)
        # print(x.shape)
        return self.encoder(x)
        # return x

class DecoderModule(nn.Module):
    def __init__(self, image_height:int, 
                 output_channel:int, 
                 encoder_output_shape:list,
                #  channel_multiply:int = 32
                 ):
        super(DecoderModule, self).__init__()

        assert image_height % encoder_output_shape[-1] == 0, f'encoder output shape cannot be upsampled by the power of 2 to match the input size {self.image_height}'
        self.num_decode_block = int( math.log10( int (image_height / encoder_output_shape[-1]) ) / math.log10(2) )

        self.output_channel = output_channel
        self.input_channel = encoder_output_shape[0]
        # self.channel_multiply = channel_multiply
        self.decoder = self._decoder()
        


    def _decoder(self):

        decoder = nn.Sequential()

        for idx in range(1, self.num_decode_block+1):

            if idx==1:
                decoder.add_module(f'decoder_block{idx}', self._decoder_block(in_channels = self.input_channel, 
                                                                            out_channels = int(self.input_channel/int(2**idx)), stride =1, dropout=0.5))
            else:
                decoder.add_module(f'decoder_block{idx}', self._decoder_block(in_channels = int(self.input_channel/int(2**(idx-1))), 
                                                                            out_channels = int(self.input_channel/int(2**idx)), stride =1, dropout= 0.5))
        decoder.add_module(f'decoder_block{self.num_decode_block+1}', nn.Conv2d(in_channels= int(self.input_channel/int(2**idx)), 
                                                                        out_channels= self.output_channel, 
                                                                        kernel_size=3, 
                                                                        stride=1, 
                                                                        padding=1, 
                                                                        bias=True))
        return decoder
    

        

    def _decoder_block(self, in_channels = 256, out_channels = 512, stride =1, dropout = 0):
        decoder_block = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False),
        nn.Dropout2d(dropout),
        ResidualBlock(in_channels, out_channels),
        nn.UpsamplingBilinear2d(scale_factor=2)
        )
        return decoder_block

    def forward(self, x):
        # print(x.shape)
        # x = self.decoder.decoder_block1(x)
        # print(x.shape)
        # x = self.decoder.decoder_block2(x)
        # print(x.shape)
        return self.decoder(x)
        # return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out

def get_pretrained_model(model_name,ImageNet, 
                         image_height:int, 
                        input_channel:int, 
                        encoder_output_shape:list,
                        output_channel:int,
                        ):
    """Retrieve a custom model or pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)
        image_height (int): image height  
        input_channel(int): input channel of the image 
        encoder_output_shape (list): encoder output shape [channel, height, width]
        output_channel (int): output channel of the reconstructed image

    Return
    --------
        model (PyTorch model): cnn
    """

    if model_name == 'symmetricEncoder':
        model = EncoderModule(image_height= image_height,
                              input_channel= input_channel,
                              encoder_output_shape= encoder_output_shape)
        
    elif model_name == 'symmetricDecoder':
        model = DecoderModule(image_height= image_height,
                              output_channel= output_channel,
                              encoder_output_shape= encoder_output_shape)
        
    return model


class CNNMAE(pl.LightningModule):
    """
            Arguments:
                required_model:str -> required encoder model from the utils.py file
                patch_size:int -> patch size of the mask
                mask_ratio:float -> masking ratio
                frozen_backbone:str -> if need to freeze the pretrained cnn encoder 
                ImageNet:bool -> if you want to initiate all the weights of ImageNet pretrained model
                encoder_output_shape:list -> shape of the tensor coming out of the 
                image_height:int -> input image height
    """
    
    def __init__(self,
                 required_model:list,
                 patch_size: int,
                 mask_ratio:float,
                 frozen_backbone:str,
                 ImageNet:bool, 
                 encoder_output_shape:list,
                 image_height:int,
                 output_channel:int = 3,
                optimizer:str = 'ADAM',
                learning_rate:float = 0.001,
                 factor:float = 0.1,
                 patience:int = 10, 
                 input_channel = 3
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.frozen_backbone = frozen_backbone
        self.encoder_output_shape = encoder_output_shape
        self.image_height = image_height
        self.optimizer, self.learning_rate, self.factor, self.patience = optimizer, learning_rate, factor, patience
        self.is_log_image = True

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # only inititiate model as classfication model for now as output [batch, sequence, last_hidden_state], later model will  loaded as [batch, sequence, sequence, sequence_length]
        self.encoder_module = get_pretrained_model(
                model_name = required_model[0],
                ImageNet= ImageNet, 
                image_height= image_height,
                input_channel= input_channel,
                output_channel= output_channel,
                encoder_output_shape= encoder_output_shape
            )
        
        
        if self.frozen_backbone:
            self.encoder_module = self.freeze_parameters(self.encoder_module)

        self.decoder_module = get_pretrained_model(
                model_name = required_model[1],
                ImageNet= ImageNet, 
                image_height= image_height,
                input_channel= input_channel,
                output_channel= output_channel,
                encoder_output_shape= encoder_output_shape
            )

        # loss function
        # self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.MSELoss()

        # self.device = torch.tensor("cuda" if torch.cuda.is_available() else "cpu")

    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    

    def _hybrid_mse_loss(self, input_img: torch.Tensor, output_img: torch.Tensor) -> torch.Tensor:
        mse_loss = (input_img - output_img) ** 2
        mse_loss = einops.rearrange(mse_loss, 'n c h w->n c (h w)').mean(dim=2)
        mse_loss = mse_loss.sum(dim=1)
        mse_loss = mse_loss.mean(dim=0)
        return mse_loss
    
   
    def _hybrid_l1_loss(self, input_img: torch.Tensor, output_img: torch.Tensor) -> torch.Tensor:
        input_ = einops.rearrange(input_img, 'n c (h p) (w q)->n (p q)(c h w)', p=14, q=14).mean(dim=2)
        output_ = einops.rearrange(output_img, 'n c (h p) (w q)->n (p q)(c h w)', p=14, q=14).mean(dim=2)
        loss = nn.L1Loss()
        return loss(input_, output_)
    
    
    def _maskedPatch_MSE_loss(self, input_img: torch.Tensor, output_img: torch.Tensor, mask: torch.tensor) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ = einops.rearrange(input_img, 'n c (h p) (w q)->n (p q)(c h w)', p= 14, q=14)    #<-- p,q value is hard coded
        output_ = einops.rearrange(output_img, 'n c (h p) (w q)->n (p q)(c h w)', p=14, q=14)   #<-- p,q value is hard coded
        loss = nn.MSELoss()
        input_ = input_[:, (mask == 1).nonzero(as_tuple=True)[1], :]
        output_ = output_[:, (mask == 1).nonzero(as_tuple=True)[1], :]
        loss_ = loss(input_, output_)
        del input_, output_
        return loss_.to(device)
    
    
    def _maskedPatch_L1_loss(self, input_img: torch.Tensor, output_img: torch.Tensor, mask: torch.tensor) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ = einops.rearrange(input_img, 'n c (h p) (w q)->n (p q)(c h w)', p= 14, q=14)    #<-- p,q value is hard coded
        output_ = einops.rearrange(output_img, 'n c (h p) (w q)->n (p q)(c h w)', p=14, q=14)   #<-- p,q value is hard coded
        loss = nn.L1Loss()
        input_ = input_[:, (mask == 1).nonzero(as_tuple=True)[1], :]
        output_ = output_[:, (mask == 1).nonzero(as_tuple=True)[1], :]
        loss_ = loss(input_, output_)
        del input_, output_
        return loss_.to(device)

       

    def patchify_image(self,
                       input_image: torch.tensor,
                        patch_size: int = 16):
        """
        input_image: torch.tensor -> image as a tensor with shape [batch, channel, height, width]
        patch_size: int = 16 -> size of the patch 
        
        """

        batch_size, channels, height, width = input_image.shape

        assert height == width, 'height and width should be equal'
        assert height % patch_size == 0, 'height and width size should be divisible by the patch size and remainder should be 0'
        
        height_patch_num = (height // patch_size)   # calculate the total number of patches in height
        width_patch_num = (width // patch_size) # calculate the total number of patches in width
        
        x = input_image.reshape(batch_size, channels, height_patch_num, patch_size, width_patch_num, patch_size)    # reshaped image to 6 dimensional tensor
        x = torch.einsum('nchpwq->nhwpqc', x)   # rearranging the tensors
        x = x.reshape(batch_size, height_patch_num*width_patch_num, patch_size**2 * channels)   # reshaped tensor to shape of [batch_size, sequence, sequence_length]
        return x

    def masked_tensor(self,
                      patchified_tensor:torch.tensor, 
                    mask_ratio:float = 0.3):
        """
        patchified_tensor:torch.tensor -> this is the patchified tensor with shape [batch_size, sequence, sequence_length]
        mask_ratio:float = 0.5 -> ratio of the mask on patches should be applied on each batch
        """
        batch_size, sequence, sequence_length = patchified_tensor.shape
        len_keep = int(sequence * (1 - mask_ratio))
        noise = torch.rand((batch_size, sequence))  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, sequence])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # generate masked image 
        patchified_tensor_ = patchified_tensor.clone()
        patchified_tensor_[mask == 1] = 0

        return patchified_tensor_, mask

    def unpatchify_tensor(self,
                          patchified_tensor:torch.tensor, 
                        patch_size: int = 16):
        """
        patchified_tensor:torch.tensor -> this is the patchified tensor with shape [batch_size, sequence, sequence_length]
        patch_size: int = 16 -> size of the patch
        """
        batch_size, sequence, sequence_length = patchified_tensor.shape
        x = patchified_tensor.reshape(batch_size, int(math.sqrt(sequence)), int(math.sqrt(sequence)), 
                                    patch_size, patch_size, int(sequence_length/(patch_size**2))) # reshaped image to 6 dimensional tensor
        x = torch.einsum('nhwpqc->nchpwq', x)   # rearranging the tensors
        x = x.reshape(batch_size, int(sequence_length/(patch_size**2)), int(math.sqrt(sequence))*patch_size, int(math.sqrt(sequence))*patch_size)   # reshaped tensor to shape of [batch_size, sequence, sequence_length]
        return x
    
    def forward(self, x):
        x = self.encoder_module(x)
        x = self.decoder_module(x)
        return x
    
    def on_train_epoch_end(self):
        self.is_log_image = True
        torch.cuda.empty_cache()

    def _common_step(self, batch, batch_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # #######################
        # print(batch)
        # batch, _ = batch
        # #########################
        patchified_tensor = self.patchify_image(input_image= batch, patch_size= self.patch_size).to('cpu')
        patchified_tensor, mask = self.masked_tensor(patchified_tensor= patchified_tensor, mask_ratio= self.mask_ratio)
        x = self.unpatchify_tensor(patchified_tensor= patchified_tensor, patch_size= self.patch_size)
        scores = self.forward(x.to(device))
        loss = self._maskedPatch_MSE_loss(scores, batch, mask)

        if self.is_log_image:
            self._log_image(batch, x, scores)
            self.is_log_image = False
        # loss = self.loss_fn(scores, batch)
        return loss, scores, mask
    
    def _log_image(self, original_img: torch.Tensor, masked_img: torch.Tensor, predicted_img: torch.Tensor) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        std = torch.tensor([0.229, 0.224, 0.225]).reshape( 3, 1, 1).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape( 3, 1, 1).to(device)

        tb_logger = self.logger.experiment
        cat_original_img = original_img[-1]*std+mean
        cat_masked_img = masked_img[-1].to(device)*std+mean
        cat_predicted_img = predicted_img[-1]*std+mean
        # cat_original_img = original_img[-1]*255.0
        # cat_masked_img = masked_img[-1].to(device)*255.0
        # cat_predicted_img = predicted_img[-1]*255.0
        cat_log_img = torch.cat([cat_original_img.to(device), cat_masked_img.to(device), cat_predicted_img.to(device)], dim=-1)

        tb_logger.add_image(f'self-supervised total episode: {self.current_epoch}',
                            cat_log_img, self.current_epoch, dataformats='CHW')
    
    def training_step(self, batch, batch_idx):
        # #######################
        # batch, _ = batch
        # #########################
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log_dict({'train_loss': loss},
                      on_step = False, on_epoch = True, prog_bar=True, logger = True)
        return {'loss': loss}
    
    
    def validation_step(self, batch, batch_idx):
        # #######################
        # batch, _ = batch
        # #########################
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log_dict({'val_loss': loss},
                      on_step = False, on_epoch = True, prog_bar=True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # #######################
        # batch, _ = batch
        # #########################
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log_dict({'test_loss': loss},
                      on_step = False, on_epoch = True, prog_bar=True, logger = True)
        return loss
    
    def predict_step(self, batch, batch_idx): 
        # #######################
        # batch, _ = batch
        # #########################
        loss, scores, mask = self._common_step(batch, batch_idx)
        return loss, scores, batch, mask
    
    def predict_image(self, batch, batch_idx):
        # #######################
        # batch, _ = batch
        # ######################### 
        loss, scores, mask = self._common_step(batch, batch_idx)
        return loss, scores, batch, mask
    
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