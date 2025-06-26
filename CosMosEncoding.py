from libraries import *
class CosmosEmbedding:
    def __init__(self):
        """
        Initializes the CosmosEmbedding class and loads the encoder model.
        """
        self.encoder_model = self.__init_encoder_model()

    def __init_encoder_model(self):
        """
        Initializes the COSMOS video tokenizer encoder.

        Returns:
            CausalVideoTokenizer: Encoder used for generating video embeddings.
        """
        model_name = "Cosmos-0.1-Tokenizer-CV8x16x16"

        # Load pre-trained encoder from local checkpoint
        encoder = CausalVideoTokenizer(
            checkpoint_enc=f'./pretrained_ckpts/{model_name}/encoder.jit',
            device="cuda",
            dtype="bfloat16"
        )
        # encoder.eval()
        return encoder

    def generate_embedding(self, input_video, temporal_window=8):
        """
        Generates embeddings from an input video using a sliding temporal window.

        Args:
            input_video (np.ndarray): Input video of shape [T, H, W, C],
                                      where T = number of frames.
            temporal_window (int): Number of frames per chunk/window used
                                   to compute each embedding.

        Returns:
            np.ndarray: Concatenated latent embeddings of shape 
                        [num_windows, latent_dim, ...]
        """
        # Add batch dimension: [T, H, W, C] → [1, T, H, W, C]
        raw_input = np.expand_dims(input_video, axis=0)
        num_frames = raw_input.shape[1]

        output_latent_list = []

        # Process video in temporal chunks
        for idx in range(0, (num_frames - 1) // temporal_window + 1):
            start = idx * temporal_window
            end = (idx + 1) * temporal_window

            # Slice out the current temporal window
            window_video = raw_input[:, start:end, ...]

            # Convert NumPy to torch.Tensor with required dtype and device
            input_tensor = numpy2tensor(
                window_video,
                dtype=torch.bfloat16,
                device=torch.device('cuda')
            )
            with torch.no_grad():
                # Encode the window using the pre-trained COSMOS encoder
                raw_latent, = self.encoder_model.encode(input_tensor)

            # Convert back to NumPy for downstream use
            raw_latent_np = tensor2numpy(raw_latent)

            # Store latent for current window
            output_latent_list.append(raw_latent_np)

        # Concatenate latents from all temporal windows along time axis
        output_latent_np = np.concatenate(output_latent_list, axis=1)

        return output_latent_np
