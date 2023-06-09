from src.models.core import CoreConfig


class NATCoreConfig(CoreConfig):

    def __init__(self,
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
                 sos_token_id: int = 0,
                 eos_token_id: int = 2,
                 pad_token_id: int = 1,
                 length_token_id: int = None,
                 label_smoothing: float = 0.0,
                 src_embedding_copy: str = None,
                 pooler_size: int = 256) -> None:
        """
        Configuration class for the GLAT model.
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
        :param sos_token_id: the start of sequence token id (default=0).
        :param eos_token_id: the end of sequence token id (default=2).
        :param pad_token_id: the pad token id (default=1).
        :param length_token_id: the length token id, akin to a cls token (default=4).
        :param label_smoothing: the label smoothing value for the cross-entropy loss (default=0.0).
        :param src_embedding_copy: the type of copy to apply to the source embedding, possible values: uniform, soft and
            None (default=None).
        :param pooler_size: the pooler layer dimension (default=256).
        """
        super().__init__(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_ff, dropout,
                         dropout_mha, dropout_ff, activation_ff, layer_norm_eps, scale_embeddings, sos_token_id,
                         eos_token_id, pad_token_id, label_smoothing)
        self.length_token_id = length_token_id
        if src_embedding_copy not in ["uniform", "soft", None]:
            raise ValueError("The source embeddings copy can only be performed with one of the following mode: uniform,"
                             "soft or None.")

        self.src_embedding_copy = src_embedding_copy
        self.pooler_size = pooler_size