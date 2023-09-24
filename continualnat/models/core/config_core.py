class CoreConfig:

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        dropout_mha: float = 0.0,
        dropout_ff: float = 0.0,
        activation_ff: str = "relu",
        layer_norm_eps: float = 1e-6,
        scale_embeddings: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Base class for the models' configurations.
        :param vocab_size: shared vocabulary size.
        :param d_model: embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param num_encoder_layers: the number of encoder layers (default=6).
        :param num_decoder_layers: the number of decoder layers (default=6).
        :param dim_ff: dimension of the feedforward sublayer (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param dropout_mha: the dropout value for the multi-head attention (default=0.0).
        :param dropout_ff: the dropout value for the feed-forward sublayer (default=0.0).
        :param activation_ff: the activation function for the feed-forward sub-layer, can be either ReLU or GeLU
            (default="relu").
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        :param scale_embeddings: whether to scale the output of the embedding layer with the inverse square root
            of d_model (default=False).
        :param bos_token_id: the start of sequence token id (default=0).
        :param eos_token_id: the end of sequence token id (default=2).
        :param pad_token_id: the pad token id (default=1).
        :param label_smoothing: the label smoothing value for the cross-entropy loss (default=0.0).
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.dropout_mha = dropout_mha
        self.dropout_ff = dropout_ff
        self.activation_ff = activation_ff
        self.layer_norm_eps = layer_norm_eps
        self.scale_embeddings = scale_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
