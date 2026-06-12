"""Frozen Transformers 5.7.0 machinery + outputs.

Initial vendored snapshot. The static-miner LANDMARKS cover the
``GenerationConfig.validate`` and ``BitsAndBytesConfig.post_init`` seams
the miner walks. The discovery LANDMARKS cover the ``from_pretrained``
signatures and ``GenerationConfig.to_dict`` schema-discovery surface.
"""
