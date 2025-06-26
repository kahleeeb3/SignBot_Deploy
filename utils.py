from libraries import *
from cnn_mae import *   
import random 
import natsort

class VideoDatasetRealTime(pl.LightningDataModule):
    """
    A real-time video dataset for loading and transforming raw and landmark videos,
    typically used for single-sample inference in video classification tasks.
    """
    def __init__(self, raw_video, 
                 landmark_video, 
                 sampling_method: str = 'uniform', 
                 num_frame: int = 20):
        """
        Initializes the video dataset.

        Args:
            raw_video (np.ndarray ): Raw video data (frames x H x W x C).
            landmark_video (np.ndarray): Corresponding landmark video.
            sampling_method (str): Method to sample frames from the video. Default is 'uniform'.
            num_frame (int): Number of frames to sample. Default is 20.
        """
        super().__init__()
        self.raw_video = raw_video
        self.landmark_video = landmark_video
        self.sampling_method = sampling_method
        self.num_frame = num_frame
    
    def __len__(self):
        """
        Returns the number of samples. Since this is likely used for a single prediction,
        the logic is a placeholder and should reflect actual use case if used in training.

        Returns:
            int: Number of samples.
        """
        return int(self.raw_video.shape[0]+self.landmark_video.shape[0]/self.landmark_video.shape[0])

    def read_video(self, video):
        """
        Converts a video (sequence of frames) to a tensor with necessary transforms.

        Args:
            video (np.ndarray): Video data as a numpy array of shape (T x H x W x C).

        Returns:
            torch.Tensor: Transformed video tensor of shape (T x C x H x W).
        """
        transform = torchvision.transforms.Compose(
            [   
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                              
            ]
        )
        frame_tensor = torch.tensor([])

        for idx in range(video.shape[0]):
            # concatenate the frame tensors
            image = Image.fromarray(video[idx, :, :, :])
            if frame_tensor.shape[0] == 0:
                # Convert the frame to a PIL image
                
                frame_tensor = transform(image).unsqueeze(0)
            else:
                frame_tensor  = torch.cat((frame_tensor, transform(image).unsqueeze(0)), dim = 0)
        return frame_tensor
    
    def fixed_frames(self, reader, random_seed):
        """
        Samples a fixed number of frames from a video using the chosen sampling method.

        Args:
            reader (torch.Tensor): Video tensor of shape (T, C, H, W).
            random_seed (int or None): Seed for random sampling.

        Returns:
            torch.Tensor: Sampled/padded video tensor of shape (num_frame, C, H, W).
            int: Used random seed.
        """
        if not self.num_frame == None:
            if reader.shape[0] < self.num_frame:
                # Pad with the last frame
                number_of_padded_timestamp = self.num_frame - reader.shape[0]
                X = torch.cat((reader, torch.cat(number_of_padded_timestamp*[reader[-1, :, :, :, ].unsqueeze(0)])), dim=0)
            elif reader.shape[0] == self.num_frame:
                X = reader
            else:
                if self.sampling_method == 'random':
                    if random_seed == int:
                        random.seed(random_seed)
                        random_frame_idx = natsort.natsorted(random.sample(range(0, reader.shape[0]), self.num_frame))
                    else:
                        random_seed = random.randint(0, 100)
                        random.seed(random_seed)
                        random_frame_idx = natsort.natsorted(random.sample(range(0, reader.shape[0]), self.num_frame))   # creating array for index of randomly taking 20 frames out of all frames available to the video
                    X = reader[random_frame_idx, :, :, :]    #video with 20 frames with our generated random index
                elif self.sampling_method == 'uniform':
                    frame_idx = np.linspace(start = 0, stop = reader.shape[0], endpoint= False, num= self.num_frame, dtype= np.int32)
                    X = reader[frame_idx, :, :, :]
        else:
            X = reader
        return X, random_seed
    
    def __getitem__(self, idx):
        """
        Returns the concatenated preprocessed and sampled raw and landmark videos.

        Args:
            idx (int): Index (ignored in real-time setting).

        Returns:
            torch.Tensor: Concatenated tensor of shape (2*num_frame, C, H, W).
        """
        
        raw_video = self.read_video(self.raw_video)
        landmark_video = self.read_video(self.landmark_video)

        raw_video_fixed, random_seed = self.fixed_frames(raw_video, random_seed='None')
        landmark_video_fixed, random_seed = self.fixed_frames(landmark_video, random_seed=random_seed)
        
        X = torch.cat((raw_video_fixed, landmark_video_fixed), dim=0)
        return X


class CosmosEmdDataset(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading latent embeddings from raw and landmark streams.
    Pads embeddings to a fixed length (150 frames) and generates a binary mask.
    """
    def __init__(self, 
               raw_latent,
               landmark_latent
               ):
        """
        Initializes the dataset with raw and landmark latent inputs.

        Args:
            raw_latent (np.ndarray): Raw latent tensor with shape (1, T, H, W, C).
            landmark_latent (np.ndarray): Landmark latent tensor with shape (1, T, H, W, C).
        """
        super().__init__()
        # Remove batch dimension (assumed to be 1)
        self.raw_latent = np.squeeze(raw_latent, axis = 0)
        self.landmark_latent = np.squeeze(landmark_latent, axis = 0)

    def __len__(self):
        # Fixed length, single-sample real-time use
        return 1
    

    def __getitem__(self, idx):
        """
        Returns a padded sample with abstract embeddings and a binary mask.

        Args:
            idx (int): Sample index (ignored in this context).

        Returns:
            tuple: ((raw_embedding, landmark_embedding), -1, mask_frame)
                   where embeddings are shape (150, C, H, W), and mask_frame is (150,)
        """
        # get the shape of the data
        latent_time, H, W, C = self.raw_latent.shape

        # generate zero tensor for mask, raw and landmark embedding
        raw_abstract_embedding = torch.zeros((150, H, W, C))
        landmark_abstract_embedding = torch.zeros((150, H, W, C))
        mask_frame = torch.zeros(150)

        # assign the embddings to the zero tensors
        raw_abstract_embedding[:latent_time] = torch.tensor(self.raw_latent[:latent_time, :, :, :])
        landmark_abstract_embedding[:latent_time] = torch.tensor(self.landmark_latent[:latent_time, :, :, :])
        mask_frame[:latent_time] = 1

        # Rearrange to (T, C, H, W)
        raw_abstract_embedding = raw_abstract_embedding.permute(0,3,1,2)
        landmark_abstract_embedding = landmark_abstract_embedding.permute(0,3,1,2)

        return (raw_abstract_embedding, landmark_abstract_embedding), -1, mask_frame




def get_pretrained_encdec(model_name, model_path):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn
    """

    if model_name == 'CNNMAE':
        "This model is trained in Masked AutoEncoder Technique. The encoder size is nearly double than decoder size. Loss was calculated as masked patch loss."
        from cnn_mae import CNNMAE
        yaml_path = os.path.join(model_path, "hparams.yaml")
        if not os.path.exists(yaml_path):

            yaml_path = os.path.join(".", 'CNNMAE_pre_trained_model_path',model_path.split(os.sep)[-1],'hparams.yaml')
        with open(yaml_path, 'r') as file:
            hparams = yaml.safe_load(file)
        check_path = os.path.join(model_path, "best_epoch.ckpt")
        if not os.path.exists(check_path):
            check_path = os.path.join(".", "CNNMAE_pre_trained_model_path",model_path.split(os.sep)[-1], "best_epoch.ckpt")
            # print(check_path)
        model = CNNMAE.load_from_checkpoint(check_path, **hparams)
        model.mask_ratio = 0
    return model

