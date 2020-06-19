        # self.layers_a = nn.ModuleList(  #Text to Audio (The query vector comes from the Text and Key-Value from the Audio)
        #     [
        #         TransformerSentenceEncoderLayer(
        #             embedding_dim=self.embedding_dim,
        #             ffn_embedding_dim=ffn_embedding_dim,
        #             num_attention_heads=num_attention_heads,
        #             dropout=self.dropout,
        #             attention_dropout=attention_dropout,
        #             activation_dropout=activation_dropout,
        #             activation_fn=activation_fn,
        #             add_bias_kv=add_bias_kv,
        #             add_zero_attn=add_zero_attn,
        #             export=export,
        #         )
        #         for _ in range(num_encoder_layers)
        #     ]
        # )

        # self.layers_v = nn.ModuleList(  #Text to Audio (The query vector comes from the Text and Key-Value from the Audio)
        #     [
        #         TransformerSentenceEncoderLayer(
        #             embedding_dim=self.embedding_dim,
        #             ffn_embedding_dim=ffn_embedding_dim,
        #             num_attention_heads=num_attention_heads,
        #             dropout=self.dropout,
        #             attention_dropout=attention_dropout,
        #             activation_dropout=activation_dropout,
        #             activation_fn=activation_fn,
        #             add_bias_kv=add_bias_kv,
        #             add_zero_attn=add_zero_attn,
        #             export=export,
        #         )
        #         for _ in range(num_encoder_layers)
        #     ]
        # )