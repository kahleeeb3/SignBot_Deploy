from libraries import *
from models import *
from utils import *

import torchmetrics
class ModelPrediction:
    """
    A class to handle video classification using a pre-trained model and make predictions
    based on softmax probability thresholds with conformal prediction.
    """
    def __init__(self,
                 model_path:str):
        self.model_path = model_path
        self.model = None
        self.__init__model()
        self.label_encode = self.label_transform()

    
    def __init__model(self):
        """
        Initializes the ModelPrediction object.

        Args:
            model_path (str): Path to the trained model checkpoint.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision('medium')
        self.model_new = VideoClassificationModel.load_from_checkpoint(self.model_path).to(device)
        self.model_new.is_log_image = False
        self.model_new.eval()

    def label_transform(self):
        """
        Sets up label encoding and decoding dictionaries.

        Returns:
            dict: A dictionary containing index-to-class mapping,
                  class-to-index mapping, and class name list.
        """

        label_encode = {'idx_2_class': {0: 'forward',
        1: 'right',
        2: 'left',
        -1 : 'Uncertain'},
        'class_2_idx': {'forward': 0,
        'right': 1,
        'left': 2,
        'Uncertain': -1},
        'class_list': ['class_01_move_ahead', 'class_02_right', 'class_03_left', 'Uncertain']}
        return label_encode
    
    def dataloader(self, raw_video, landmark_video):
        """
        Loads a single sample from the video dataset for inference.

        Args:
            raw_video: Raw video input (e.g., RGB frames).
            landmark_video: Corresponding landmark video data.

        Returns:
            Tensor: A single batch (1 sample) of processed video data.
        """
        test_dataset = VideoDatasetRealTime(
        raw_video = raw_video,
        landmark_video = landmark_video,
        sampling_method = 'uniform',
        num_frame = 20
        )
        test_dataloader = torch.utils.data.DataLoader(
                                    test_dataset,
                                    batch_size = 1, 
                                    shuffle= True
                                
                                )
        test_feature = next(iter(test_dataloader))
        print('Observation shape: ', test_feature.shape)
        return test_feature
    
    def cp(self, prediction_prob):
        """
        Applies conformal prediction thresholding using a precomputed q̂ value.

        Args:
            prediction_prob (ndarray): Softmax probability scores.

        Returns:
            ndarray: Boolean array indicating which classes are selected.
        """
        with open(os.path.join(".", "cal_qhat", "alpha_0.05.npy"), 'rb') as file_:
            qhat = np.load(file_)

        # Create prediction set using (1 - q̂) threshold
        prediction_set = prediction_prob> (1-qhat)
        return prediction_set 

    
    def prediction(self, raw_video, landmark_video):
        """
        Makes a prediction on the input video using the loaded model.

        Args:
            raw_video: Raw video input.
            landmark_video: Corresponding landmark video data.

        Returns:
            str: The predicted direction label ('forward', 'right', 'left', or 'Uncertain').
        """
        test_feature = self.dataloader(raw_video, landmark_video)
        with torch.no_grad():
            # Move test input to GPU for inference
            scores = self.model_new(test_feature.to('cuda')).to('cpu')
            prediction_prob = torch.nn.functional.softmax(scores, dim=1)
            prediction_prob = prediction_prob.numpy()
            prediction_set = self.cp(prediction_prob=prediction_prob)
            # Determine the final prediction
            if np.sum(prediction_set) == 1:
                index = np.where(prediction_set)[-1][0]
                direction = self.label_encode['idx_2_class'][index]
            else:
                direction = self.label_encode['idx_2_class'][-1]
        return direction
    
class CosmosModelPrediction:
    """
    Class for making predictions using a pretrained Cosmos model
    that processes latent embeddings (e.g., from vision data).

    Handles loading the model, preparing input data, and returning predicted labels.
    """
    def __init__(self,
                 model_path:str):
        """
        Initializes the model prediction class by loading the model and label encoder.

        Args:
            model_path (str): Path to the trained model checkpoint file.
        """
        self.model_path = model_path
        self.model = None
        self.__init__model()
        self.label_encode = self.label_transform()

    
    def __init__model(self):
        """
        Internal method to load the trained Cosmos model and set it to evaluation mode.
        Automatically selects CUDA if available.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision('medium')
        self.model_new = Cosmos_V2_Weighted_Loss.load_from_checkpoint(self.model_path).to(device)
        self.model_new.eval()

    def label_transform(self):
        """
        Loads the label encoder dictionary from a pickled file.

        Returns:
            dict: A dictionary mapping between label indices and class names.
        """
        import pickle
        label_encode_path = os.path.join(os.path.dirname(self.model_path), 'label_encoder.pkl')
        with open(label_encode_path, 'rb') as f:
            label_encode = pickle.load(f)

        return label_encode
    
    def dataloader(self, raw_latent, landmark_latent):
        """
        Prepares the data loader for inference using latent feature inputs.

        Args:
            raw_latent: Raw latent features from video/image input.
            landmark_latent: Landmark-based latent features.

        Returns:
            tuple: A batch from the dataset containing abstract embeddings and mask frames.
        """
        test_dataset = CosmosEmdDataset(
        raw_latent,
        landmark_latent
        )
        test_dataloader = torch.utils.data.DataLoader(
                                    test_dataset,
                                    batch_size = 1, 
                                    shuffle= True
                                
                                )
        test_feature = next(iter(test_dataloader))
        return test_feature

    
    def prediction(self, raw_latent, landmark_latent):
        """
        Performs prediction on the given input data using the loaded model.

        Args:
            raw_latent: Raw latent embeddings.
            landmark_latent: Landmark latent embeddings.

        Returns:
            str: Predicted class label.
        """
        # Prepare the input feature from data loader
        test_feature = self.dataloader(raw_latent, landmark_latent)
        [abstract_embedding, _, mask_frame] = test_feature
        with torch.no_grad():
            # Forward pass through the model
            scores = self.model_new.forward((abstract_embedding[0].to('cuda'), abstract_embedding[1].to('cuda')), mask_frame.to('cuda'))
            # Convert to probabilities and find predicted index
            
            prediction_prob = torch.nn.functional.softmax(scores, dim=1)
            print(f'Model probabilities {prediction_prob}')
            index = torch.argmax(scores, 1).cpu().detach().item()
            # Get label from encoder
            direction = self.label_encode['idx_2_class'][index]

        return direction